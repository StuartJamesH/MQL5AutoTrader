//+------------------------------------------------------------------+
//|                                               DonchianTrend.mq5 |
//| Donchian channel trend gate                                      |
//|                                                                  |
//| Replicates donchian_trend() from                                 |
//| ModelWorkbench/Learn/features.py as visualised in               |
//| ModelWorkbench/1_1 Signal Lab.ipynb                              |
//|                                                                  |
//| Algorithm                                                        |
//|   hh[i] = rolling max of High over `length` bars                |
//|   ll[i] = rolling min of Low  over `length` bars                |
//|   trend:                                                         |
//|     if Close[i] > hh[i-1]  →  +1  (uptrend)                    |
//|     if Close[i] < ll[i-1]  →  -1  (downtrend)                  |
//|     else                   →  carry previous trend              |
//|                                                                  |
//| Colour scheme (matches RegimeMA indicator)                       |
//|   Uptrend   (#26a69a teal)                                       |
//|   Downtrend (#ef5350 red)                                        |
//|   Neutral   (#888888 gray) — startup bars before channel fills   |
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "1.00"
#property indicator_separate_window
#property indicator_minimum  -1.5
#property indicator_maximum   1.5
#property indicator_buffers  4
#property indicator_plots    3

//--- Uptrend  (+1, teal #26a69a)
#property indicator_label1  "Uptrend"
#property indicator_type1   DRAW_LINE
#property indicator_color1  C'38,166,154'
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- Downtrend (-1, red #ef5350)
#property indicator_label2  "Downtrend"
#property indicator_type2   DRAW_LINE
#property indicator_color2  C'239,83,80'
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- Neutral (0, gray #888888) — carries during startup
#property indicator_label3  "Neutral"
#property indicator_type3   DRAW_LINE
#property indicator_color3  C'136,136,136'
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

//--- Input
input int InpLength = 60;  // Donchian Channel Length (bars)

//--- Display buffers
double UpBuffer[];      // value = +1 when trend is up,   else EMPTY_VALUE
double DownBuffer[];    // value = -1 when trend is down, else EMPTY_VALUE
double NeutralBuffer[]; // value =  0 during startup,     else EMPTY_VALUE

//--- Calculation buffer
double TrendBuffer[];   // stored trend: +1 / -1 / 0  (persists across ticks)

//+------------------------------------------------------------------+
int OnInit()
{
    SetIndexBuffer(0, UpBuffer,      INDICATOR_DATA);
    SetIndexBuffer(1, DownBuffer,    INDICATOR_DATA);
    SetIndexBuffer(2, NeutralBuffer, INDICATOR_DATA);
    SetIndexBuffer(3, TrendBuffer,   INDICATOR_CALCULATIONS);

    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);

    // Don't draw until the channel has fully populated
    PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpLength);
    PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, InpLength);
    PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, InpLength);

    // Horizontal reference lines at +1 and -1
    IndicatorSetInteger(INDICATOR_LEVELS,      2);
    IndicatorSetDouble (INDICATOR_LEVELVALUE,  0, 1.0);
    IndicatorSetDouble (INDICATOR_LEVELVALUE,  1, -1.0);
    IndicatorSetInteger(INDICATOR_LEVELSTYLE,  0, STYLE_DOT);
    IndicatorSetInteger(INDICATOR_LEVELSTYLE,  1, STYLE_DOT);
    IndicatorSetInteger(INDICATOR_LEVELCOLOR,  0, C'38,166,154');
    IndicatorSetInteger(INDICATOR_LEVELCOLOR,  1, C'239,83,80');

    IndicatorSetString(INDICATOR_SHORTNAME,
                       StringFormat("Donchian Trend(%d)", InpLength));

    return INIT_SUCCEEDED;
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
    if(rates_total < InpLength + 1) return 0;

    // Oldest-first indexing (index 0 = oldest bar)
    ArraySetAsSeries(close, false);
    ArraySetAsSeries(high,  false);
    ArraySetAsSeries(low,   false);

    // Seed bar 0 on full recalc
    int start = prev_calculated - 1;
    if(start < 1)
    {
        TrendBuffer[0]   = 0.0;
        UpBuffer[0]      = EMPTY_VALUE;
        DownBuffer[0]    = EMPTY_VALUE;
        NeutralBuffer[0] = EMPTY_VALUE;
        start = 1;
    }

    for(int i = start; i < rates_total; i++)
    {
        // Rolling Donchian channel at bar i-1 (previous bar's completed channel)
        // Compare Close[i] against the channel that was valid at bar i-1
        double hh_prev = DBL_MIN;
        double ll_prev = DBL_MAX;

        // The channel at i-1 covers bars [i-length .. i-1]
        int chan_start = i - InpLength;
        if(chan_start < 0) chan_start = 0;

        for(int k = chan_start; k < i; k++)
        {
            if(high[k] > hh_prev) hh_prev = high[k];
            if(low[k]  < ll_prev) ll_prev  = low[k];
        }

        // Trend logic (matches Python: uses previous bar's channel)
        int trend;
        bool channel_ready = (i >= InpLength); // need at least `length` prior bars

        if(!channel_ready)
        {
            trend = (int)TrendBuffer[i - 1]; // carry until channel fills
        }
        else if(close[i] > hh_prev)
        {
            trend = 1;
        }
        else if(close[i] < ll_prev)
        {
            trend = -1;
        }
        else
        {
            trend = (int)TrendBuffer[i - 1]; // carry previous trend
        }

        TrendBuffer[i] = (double)trend;

        // Route to the appropriate colour buffer
        UpBuffer[i]      = EMPTY_VALUE;
        DownBuffer[i]    = EMPTY_VALUE;
        NeutralBuffer[i] = EMPTY_VALUE;

        if(trend == 1)       UpBuffer[i]      = 1.0;
        else if(trend == -1) DownBuffer[i]    = -1.0;
        else                 NeutralBuffer[i] = 0.0;
    }

    return rates_total;
}
