// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;

using System.Data;

namespace Regression
{
    internal class LinearRegression
    {
        public List<double>? beta{get;set;}

        public static List<double> Generate_array(int n){
            List<double> array=[];
            Random rd=new(new DateTime().Microsecond);
            for(int i=0;i<n;i++)
                array.Add(rd.NextDouble());
            return array;
        }


        public static List<List<double>> Generate_Matrix(int n,int m){
            List<List<double>> X=[];
            for (int i = 0; i < n; i++)
            {
                X.Add(Generate_array(m));
            }
            return X;
        }

        public void Fit(List<List<double>> X,List<double> Y, double learning_rate, int max_iter=50,int batch_size=1,double precision=0.000001)
        {
            beta= Generate_array(X[0].Count +1);
            beta.ForEach(t=>System.Console.WriteLine(t));
            List<double> grad=[];
            // beta= Update(beta,grad,learning_rate);
            // double norme=Norme(grad);
            int k=0;
            while (k<max_iter) //norme>precision || 
            {
                for (int i = 0; i < X.Count; i++)
                {
                    grad=Gradient(X[i],Y[i]);
                    beta= Update(beta,grad,learning_rate);
                    // norme=Norme(grad);
                }
                System.Console.WriteLine($"Lost_iter({k})={Lost(X,Y)}");
                k++;
            }

        }

        private static List<double> Update(List<double> beta, List<double> grad, double learning_rate)
        {
            List<double> b=[];
            for (int i = 0; i < beta.Count; i++)
            {
                b.Add(beta[i]-learning_rate*grad[i]);
            }
            return b;
        }

        public List<double> Gradient(List<double> x,double y)
        {
            double coef = (-2.0f) * (y - Helper(x));
            List<double> grad = [coef];
            for (int i = 0; i < x.Count; i++)
            {
                grad.Add(coef * x[i]);
            }
            return grad;
        }

        private static double Norme(List<double> x)
        {
            double result = 0.0f;
            for (int i = 0; i < x.Count; i++)
            {
                result +=Math.Pow(x[i], 2.0);
            }
            return Math.Sqrt(result);
        }

        public double Lost(List<List<double>> X,List<double> Y){
            
            double l=0.0;
            for (int i = 0; i < X.Count; i++)
            {
                l+=Math.Pow(Y[i]-Helper(X[i]),2.0);
            }
            return l/X.Count;
        }

        public double Predict(List<double> x)
        {
            return Helper(x);
        }

        private double Helper(List<double> x)
        {
            double result = beta![0];
            for(int i = 0; i < x.Count; i++)
            {
                result+= x[i] * beta[i+1];
            }
            return result;
        }
    }
}
