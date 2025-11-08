// Type declarations for react-plotly.js
// This file provides TypeScript support for react-plotly.js

declare module 'react-plotly.js' {
  import { Component } from 'react';
  import { PlotParams } from 'plotly.js';

  export interface PlotProps extends Partial<PlotParams> {
    data: Partial<Plotly.PlotData>[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    frames?: Partial<Plotly.Frame>[];
    revision?: number;
    onInitialized?: (figure: Readonly<{data: Plotly.PlotData[], layout: Partial<Plotly.Layout>}>, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: Readonly<{data: Plotly.PlotData[], layout: Partial<Plotly.Layout>}>, graphDiv: HTMLElement) => void;
    onPurge?: (figure: Readonly<{data: Plotly.PlotData[], layout: Partial<Plotly.Layout>}>, graphDiv: HTMLElement) => void;
    onError?: (err: Readonly<{data: Plotly.PlotData[], layout: Partial<Plotly.Layout>}>) => void;
    // Eventi aggiuntivi supportati da Plotly, utili per interazioni con la legenda
    onLegendClick?: (event: any) => boolean | void;
    onLegendDoubleClick?: (event: any) => boolean | void;
    divId?: string;
    className?: string;
    style?: React.CSSProperties;
    debug?: boolean;
    useResizeHandler?: boolean;
  }

  export default class Plot extends Component<PlotProps> {}
}

declare global {
  namespace Plotly {
    interface PlotData {
      x?: any[];
      y?: any[];
      z?: any[];
      type?: string;
      mode?: string;
      marker?: any;
      line?: any;
      name?: string;
      text?: string | string[];
      textposition?: string;
      hovertemplate?: string;
      hoverinfo?: string;
      [key: string]: any;
    }

    interface Layout {
      title?: string | { text: string; [key: string]: any };
      xaxis?: any;
      yaxis?: any;
      width?: number;
      height?: number;
      margin?: any;
      font?: any;
      paper_bgcolor?: string;
      plot_bgcolor?: string;
      showlegend?: boolean;
      [key: string]: any;
    }

    interface Config {
      displayModeBar?: boolean;
      responsive?: boolean;
      displaylogo?: boolean;
      modeBarButtonsToRemove?: string[];
      [key: string]: any;
    }

    interface Frame {
      name?: string;
      data?: Partial<PlotData>[];
      layout?: Partial<Layout>;
      [key: string]: any;
    }
  }
}
