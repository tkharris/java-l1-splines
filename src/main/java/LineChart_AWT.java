import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;

public class LineChart_AWT extends ApplicationFrame {

    private XYSeriesCollection dataset = new XYSeriesCollection();
    private int[] x;
    private String chartTitle;

    private JFreeChart lineChart = null;
    
    public LineChart_AWT( String applicationTitle , String chartTitle , int[] x) {
      super(applicationTitle);
      this.chartTitle = chartTitle;
      this.x = x;
   }

    public void addSeries(String name, double[] vals) {
      final XYSeries s1 = new XYSeries(name);
      for(int i=0; i<x.length; i++) {
	  s1.add(x[i], vals[i]);
      }
      dataset.addSeries(s1);
    }

    public void build() {
      lineChart = ChartFactory.createXYLineChart(
         chartTitle,
         "Year","Temp",
	 dataset,
         PlotOrientation.VERTICAL,
         true,true,false);
         
      ChartPanel chartPanel = new ChartPanel( lineChart );
      chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
      setContentPane( chartPanel );
    }

    public void saveChart(String fn) throws IOException {
	ChartUtils.saveChartAsPNG(new File(fn), lineChart, 600, 400);
    }
}
