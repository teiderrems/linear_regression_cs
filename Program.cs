using System.Diagnostics;

namespace Regression
{
    internal class Program
    {

        static Mutex mutex=new();
        static void Main(string[] args)
        {
            int n = 10, m = 1;

            List<List<double>> X = Transpose(Linspace(0, 1, n, true));//LinearRegression.Generate_Matrix(n,m);
            ScottPlot.Plot plot = new();
            List<double> Y=Linspace(0, 1, n, true);//LinearRegression.Generate_array(n);
            
            double[] noise=GenerateGaussianNoise(Y.Count,0,1);
            for (int i = 0; i < Y.Count; i++)
            {
                Y[i] += noise[i];
            }
            plot.Add.Scatter(ExtractVector(X, 0), Y);
            LinearRegression lr = new();
            lr.Fit(ref X,ref Y, 0.005, 1000, 2);
            lr.beta!.ForEach(t => System.Console.WriteLine(t));
            List<double> predictions=[];
            for (int i = 0; i < Y.Count; i++)
            {
                predictions.Add(lr.Predict(X[i]));
            }
            System.Console.WriteLine("prédictions");
            predictions.ForEach(v=>System.Console.WriteLine(v));
            System.Console.WriteLine("train_label");

            Y.ForEach(v=>System.Console.WriteLine(v));
            System.Console.WriteLine("train_X");

            X.ForEach(v=>System.Console.WriteLine(v[0]));
            plot.Add.Scatter(ExtractVector(X, 0), predictions);
            plot.SavePng("./predictions.png", 400, 400);
            
        }


        private static List<List<double>> Generate_Matrix(int n,int m,int start,int end){
          List<List<double>> X=[];
          for (int i = 0; i < n; i++)
          {
            X[i]=Linspace(start,end,m);
          }
          return X;
        }



        private static List<double> ExtractVector(List<List<double>> X, int j)
        {
            List<double> column = [];
            for (int i = 0; i < X.Count; i++)
            {
                column.Add(X[i][j]);
            }
            return column;
        }

        private static List<double> Linspace(double start, double end, int num, bool endpoint = false)
        {
            Stopwatch stopwatch = new();
            stopwatch.Start();
            List<double> array = [];
            double step;
            if (endpoint)
            {
                step = (end - start) / (num - 1);
            }
            else
            {
                step = (end - start) / num;
            }
            for (int i = 0; i < num; i++)
            {
                array.Add(start + step * i);
            }
            stopwatch.Stop();
            System.Console.WriteLine($"Sequentiel_time={stopwatch.ElapsedMilliseconds} ms");
            return array;
        }

        private static List<double> Linspace_Parallele(double start, double end, int num,int nb_Thread=1, bool endpoint = false)
        {
            Stopwatch stopwatch = new();
            stopwatch.Start();
            List<double> array = [];
            double step;
            if (endpoint)
            {
                step = (end - start) / (num - 1);
            }
            else
            {
                step = (end - start) / num;
            }
            Action<object> func=(object j)=>{
                for (int i = (int) j; i < num; i+=nb_Thread)
                {
                    // mutex.WaitOne();
                    array.Add(start + step * i);

                }
            };

            List<Thread> threads=[];
            for (int i = 0; i < nb_Thread; i++)
            {
                Thread t=new(new ParameterizedThreadStart(func));
                threads.Add(t);
            }

            for (int i = 0; i < nb_Thread; i++)
            {
               threads[i].Start(i);
               threads[i].Join();
            }
            stopwatch.Stop();
            System.Console.WriteLine($"Parallele_time={stopwatch.ElapsedMilliseconds} ms");
            return array;
        }

        private static List<List<double>> Transpose(List<double> X)
        {
            List<List<double>> result = [];
            for (int i = 0; i < X.Count; i++)
            {
                result.Add([X[i]]);
            }
            return result;
        }


        static double[] GenerateGaussianNoise(int sampleCount, double mean, double stdDev)
        {
            Random random = new();
            double[] noise = new double[sampleCount];

            for (int i = 0; i < sampleCount; i += 2)
            {
                // Méthode de Box-Muller pour générer deux nombres gaussiens
                double u1 = random.NextDouble();  // Premier nombre aléatoire entre 0 et 1
                double u2 = random.NextDouble();  // Deuxième nombre aléatoire entre 0 et 1

                // Transformation Box-Muller
                double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                double z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

                // Ajustement avec la moyenne et l'écart-type
                noise[i] = mean + z0 * stdDev;

                // Si on a un nombre d'échantillons impair, il faut s'assurer que l'index est dans les limites
                if (i + 1 < sampleCount)
                {
                    noise[i + 1] = mean + z1 * stdDev;
                }
            }

            return noise;
        }
    }
}
