//+------------------------------------------------------------------+
//|                                                     RegimeMA.mq5 |
//| Causal market-regime indicator with colour-coded EMA             |
//|                                                                  |
//| Replicates causal_market_regime() from                           |
//| ModelWorkbench/Learn/labels.py as visualised in                  |
//| ModelWorkbench/1_1 Signal Lab.ipynb                              |
//|                                                                  |
//| Algorithm                                                        |
//|   1. EMA(Close, ma_period)                                       |
//|   2. slope = EMA.diff() / EMA   (price-scale invariant)         |
//|   3. Ehlers Super Smoother (2-pole IIR) applied to slope         |
//|   4. Raw regime: sign of smoothed slope (1 / -1 / 0)            |
//|   5. Forward-fill zeros with last non-zero regime                |
//|   6. Min-duration gate  (default 0 = off)                       |
//|   7. ATR percentile gate (default 0.0 = off)                    |
//|   8. Adaptive slope percentile + fixed threshold gate            |
//|                                                                  |
//| Colour scheme (matches Signal Lab plots)                         |
//|   Uptrend   #26a69a (teal)                                       |
//|   Downtrend #ef5350 (red)                                        |
//|   Range     #888888 (gray)                                       |
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 8
#property indicator_plots   3

//--- Uptrend (teal #26a69a)
#property indicator_label1  "Uptrend"
#property indicator_type1   DRAW_LINE
#property indicator_color1  C'38,166,154'
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- Downtrend (red #ef5350)
#property indicator_label2  "Downtrend"
#property indicator_type2   DRAW_LINE
#property indicator_color2  C'239,83,80'
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- Range (gray #888888)
#property indicator_label3  "Range"
#property indicator_type3   DRAW_LINE
#property indicator_color3  C'136,136,136'
#property indicator_style3  STYLE_SOLID
#property indicator_width3  2

//--- Inputs (defaults match Signal Lab regime_params)
input int    InpMAPeriod          = 60;    // EMA Period
input int    InpSlopeSmoothness   = 50;    // Super Smoother Period (slope)
input int    InpRegimeMinDuration = 0;     // Min Regime Duration bars (0 = off)
input int    InpATRWindow         = 60;    // ATR Period
input int    InpATRLookback       = 720;   // ATR Lookback for Percentile Gate
input double InpATRPercentile     = 0.0;   // ATR Percentile Gate (0 = off)
input double InpSlopeThreshold    = 5e-6;  // Fixed Slope Magnitude Threshold
input int    InpSlopeLookback     = 200;   // Adaptive Slope Lookback
input double InpSlopePercentile   = 30.0;  // Adaptive Slope Percentile

//--- Display buffers (one per regime colour)
double UpBuffer[];
double DownBuffer[];
double RangeBuffer[];

//--- Calculation buffers (persist across incremental ticks)
double EmaBuffer[];           // EMA of Close
double SlopeBuffer[];         // normalised slope: EMA.diff() / EMA
double SlopeSmoothBuffer[];   // Ehlers Super Smoother applied to slope
double ATRBuffer[];           // Wilder ATR
double RegimeFFBuffer[];      // forward-filled raw regime (1 / 0 / -1)

//+------------------------------------------------------------------+
int OnInit()
{
    SetIndexBuffer(0, UpBuffer,          INDICATOR_DATA);
    SetIndexBuffer(1, DownBuffer,        INDICATOR_DATA);
    SetIndexBuffer(2, RangeBuffer,       INDICATOR_DATA);
    SetIndexBuffer(3, EmaBuffer,         INDICATOR_CALCULATIONS);
    SetIndexBuffer(4, SlopeBuffer,       INDICATOR_CALCULATIONS);
    SetIndexBuffer(5, SlopeSmoothBuffer, INDICATOR_CALCULATIONS);
    SetIndexBuffer(6, ATRBuffer,         INDICATOR_CALCULATIONS);
    SetIndexBuffer(7, RegimeFFBuffer,    INDICATOR_CALCULATIONS);

    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);

    // Don't draw before enough history exists for the adaptive slope gate
    int draw_begin = InpMAPeriod + InpSlopeLookback;
    PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, draw_begin);
    PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, draw_begin);
    PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, draw_begin);

    IndicatorSetString(INDICATOR_SHORTNAME,
                       StringFormat("Regime MA(%d,%d)", InpMAPeriod, InpSlopeSmoothness));
    IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Compute the `percentile`-th quantile of |SlopeSmoothBuffer| over |
//| the last `window` values ending at index `idx` (oldest-first).  |
//+------------------------------------------------------------------+
double AdaptiveSlopeThreshold(const double &buf[], int idx, int window, double percentile)
{
    int count = MathMin(window, idx + 1);
    if(count <= 0) return 0.0;

    double tmp[];
    ArrayResize(tmp, count);
    for(int j = 0; j < count; j++)
        tmp[j] = MathAbs(buf[idx - count + 1 + j]);

    ArraySort(tmp);  // ascending

    int k = (int)MathFloor((count - 1) * percentile / 100.0);
    k = MathMax(0, MathMin(k, count - 1));
    return tmp[k];
}

//+------------------------------------------------------------------+
//| Compute the `percentile`-th quantile of ATRBuffer over the last  |
//| `window` values ending at index `idx` (oldest-first).           |
//+------------------------------------------------------------------+
double ATRPercentileThreshold(const double &buf[], int idx, int window, double percentile)
{
    int count = MathMin(window, idx + 1);
    if(count <= 0) return 0.0;

    double tmp[];
    ArrayResize(tmp, count);
    for(int j = 0; j < count; j++)
        tmp[j] = buf[idx - count + 1 + j];

    ArraySort(tmp);

    int k = (int)MathFloor((count - 1) * percentile / 100.0);
    k = MathMax(0, MathMin(k, count - 1));
    return tmp[k];
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double   &open[],
                const double   &high[],
                const double   &low[],
                const double   &close[],
                const long     &tick_volume[],
                const long     &volume[],
                const int      &spread[])
{
    if(rates_total < 3) return 0;

    // Use oldest-first indexing (index 0 = oldest bar) throughout
    ArraySetAsSeries(close, false);
    ArraySetAsSeries(high,  false);
    ArraySetAsSeries(low,   false);

    // ── Ehlers Super Smoother (2-pole IIR) coefficients ──────────────
    double a1    = MathExp(-1.414 * M_PI / InpSlopeSmoothness);
    double ss_c2 = 2.0 * a1 * MathCos(1.414 * M_PI / InpSlopeSmoothness);
    double ss_c3 = -(a1 * a1);
    double ss_c1 = 1.0 - ss_c2 - ss_c3;

    double ema_alpha = 2.0 / (InpMAPeriod + 1.0);

    // ── Seed bar 0 on a full recalc ───────────────────────────────────
    int start = prev_calculated - 1;
    if(start < 1)
    {
        EmaBuffer[0]         = close[0];
        SlopeBuffer[0]       = 0.0;
        SlopeSmoothBuffer[0] = 0.0;
        ATRBuffer[0]         = high[0] - low[0];
        RegimeFFBuffer[0]    = 0.0;
        UpBuffer[0]          = EMPTY_VALUE;
        DownBuffer[0]        = EMPTY_VALUE;
        RangeBuffer[0]       = EMPTY_VALUE;
        start = 1;
    }

    // ── Main bar loop (oldest → newest) ──────────────────────────────
    for(int i = start; i < rates_total; i++)
    {
        // 1. EMA
        EmaBuffer[i] = ema_alpha * close[i] + (1.0 - ema_alpha) * EmaBuffer[i - 1];

        // 2. Normalised slope: diff(EMA) / EMA[i]  (matches Python: ma.diff() / ma)
        SlopeBuffer[i] = (EmaBuffer[i] > 0.0)
                         ? (EmaBuffer[i] - EmaBuffer[i - 1]) / EmaBuffer[i]
                         : 0.0;

        // 3. Ehlers Super Smoother (2-pole IIR)
        //    filt[0]  = s[0]
        //    filt[1]  = c1*(s[1]+s[0])/2 + c2*filt[0]
        //    filt[i]  = c1*(s[i]+s[i-1])/2 + c2*filt[i-1] + c3*filt[i-2]
        if(i == 1)
            SlopeSmoothBuffer[i] = ss_c1 * (SlopeBuffer[i] + SlopeBuffer[i - 1]) / 2.0
                                 + ss_c2 * SlopeSmoothBuffer[i - 1];
        else
            SlopeSmoothBuffer[i] = ss_c1 * (SlopeBuffer[i] + SlopeBuffer[i - 1]) / 2.0
                                 + ss_c2 * SlopeSmoothBuffer[i - 1]
                                 + ss_c3 * SlopeSmoothBuffer[i - 2];

        // 4. Wilder ATR (matches talib ATR)
        //    Warm-up: running SMA; after InpATRWindow bars → Wilder's smoothing
        double tr = MathMax(high[i] - low[i],
                   MathMax(MathAbs(high[i] - close[i - 1]),
                           MathAbs(low[i]  - close[i - 1])));
        if(i <= InpATRWindow)
            ATRBuffer[i] = (ATRBuffer[i - 1] * (i - 1) + tr) / i;
        else
            ATRBuffer[i] = (ATRBuffer[i - 1] * (InpATRWindow - 1) + tr) / InpATRWindow;

        double ss_val = SlopeSmoothBuffer[i];

        // 5. Raw regime from sign of smoothed slope
        int regime_raw = (ss_val > 0.0) ? 1 : (ss_val < 0.0) ? -1 : 0;

        // 6. Forward-fill: carry last non-zero regime through flat patches
        int regime_ff;
        if(regime_raw != 0)
            regime_ff = regime_raw;
        else
            regime_ff = (int)RegimeFFBuffer[i - 1];
        RegimeFFBuffer[i] = (double)regime_ff;

        // 7. Minimum duration gate
        //    Require regime_min_duration consecutive bars of the same regime;
        //    otherwise classify as Range. Disabled when InpRegimeMinDuration <= 1.
        int regime_final = regime_ff;
        if(InpRegimeMinDuration > 1)
        {
            int min_run = InpRegimeMinDuration - 1;
            int run = 0;
            for(int k = i; k >= i - min_run && k >= 0; k--)
            {
                if((int)RegimeFFBuffer[k] == regime_ff) run++;
                else { run = 0; break; }
            }
            if(run <= min_run) regime_final = 0;
        }

        // 8. ATR percentile gate (disabled by default: InpATRPercentile == 0.0)
        //    Bars where ATR <= the Nth percentile of its own history are Range.
        if(InpATRPercentile > 0.0 && i >= InpATRLookback)
        {
            double atr_thresh = ATRPercentileThreshold(ATRBuffer, i,
                                                       InpATRLookback,
                                                       InpATRPercentile);
            if(ATRBuffer[i] <= atr_thresh)
                regime_final = 0;
        }

        // 9. Flat-slope gate
        //    a) Adaptive: |slope_sm| < rolling percentile of past |slope_sm|
        //    b) Fixed:    |slope_sm| < InpSlopeThreshold
        //    Either condition marks the bar as Range.
        bool flat_slope = false;
        if(i >= InpSlopeLookback)
        {
            double adaptive_thresh = AdaptiveSlopeThreshold(SlopeSmoothBuffer, i,
                                                            InpSlopeLookback,
                                                            InpSlopePercentile);
            flat_slope = (MathAbs(ss_val) < adaptive_thresh);
        }
        if(InpSlopeThreshold > 0.0)
            flat_slope = flat_slope || (MathAbs(ss_val) < InpSlopeThreshold);
        if(flat_slope) regime_final = 0;

        // 10. Assign EMA value to the appropriate colour buffer
        UpBuffer[i]    = EMPTY_VALUE;
        DownBuffer[i]  = EMPTY_VALUE;
        RangeBuffer[i] = EMPTY_VALUE;

        if(regime_final == 1)       UpBuffer[i]    = EmaBuffer[i];
        else if(regime_final == -1) DownBuffer[i]  = EmaBuffer[i];
        else                        RangeBuffer[i] = EmaBuffer[i];
    }

    return rates_total;
}
