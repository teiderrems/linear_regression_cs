namespace Regression
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int n=10,m=1;

            List<List<double>> X=LinearRegression.Generate_Matrix(n,m);
            List<double> Y=LinearRegression.Generate_array(n);
            LinearRegression lr=new();
            lr.Fit(X,Y,0.0001,1000);
            lr.beta!.ForEach(t=>System.Console.WriteLine(t));

            Console.WriteLine($"Hello, World !");
        }
    }
}
