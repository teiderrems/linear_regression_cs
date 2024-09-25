namespace Regression
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int n=12,m=1;

            List<List<double>> X=LinearRegression.Generate_Matrix(n,m);
            ScottPlot.Plot plot=new();
            List<double> Y=LinearRegression.Generate_array(n);
            plot.Add.Scatter(ExtractVector(X,0),Y);
            plot.SavePng("./file.png",400,400);
            LinearRegression lr=new();
            lr.Fit(X,Y,0.005,10000,1);
            lr.beta!.ForEach(t=>System.Console.WriteLine(t));
        }

        private static List<double> ExtractVector(List<List<double>> X,int j){
            List<double> column=[];
            for (int i = 0; i < X.Count; i++)
            {
                column.Add(X[i][j]);
            }
            return column;
        }
    }
}
