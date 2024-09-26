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
        private static readonly Random rd=new(new DateTime().Microsecond);
        public List<double>? beta{get;set;}

        public static List<double> Generate_array(int n){
            List<double> array=[];
            
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

        public static List<List<double>> Add_Rows(List<List<double>> X,List<double> T,int n){
            while (X.Count%n!=0)
            {
                X.Add(Generate_array(X[0].Count));
                T.Add(rd.NextDouble());
            }
            return X;
        } 


        private  static List<List<double>> Extract_Batch(List<List<double>> X,int start, int batch_size){
            List<List<double>> batch=[];
            for (int i = start; i < start+batch_size; i++)
            {
                batch.Add(X[i]);
            }
            return batch;
        }


        public void Fit(ref List<List<double>> X,ref List<double> Y, double learning_rate, int max_iter=50,int batch_size=1,double precision=0.000001)
        {
            beta= Generate_array(X[0].Count +1);
            beta.ForEach(t=>System.Console.WriteLine(t));
            List<double> grad=[];
            if (X.Count%batch_size!=0)
            {
                X=Add_Rows(X,Y,batch_size);
            }
            int k=0;
            while (k<max_iter) //norme>precision || 
            {
                if (batch_size==1)
                {
                    for (int i = 0; i < X.Count; i++)
                    {
                        grad=Gradient(X[i],Y[i]);
                        beta= Update(beta,grad,learning_rate);
                    }
                }
                else if (batch_size>1)
                {
                    for(int i=0;i<X.Count; i+=batch_size){
                        List<List<double>> batch=Extract_Batch(X,i,batch_size);
                        List<double> T=Extract_Target(Y,i,batch_size);
                        grad=Gradient(batch,T);
                        beta=Update(beta,grad,learning_rate);
                    }
                }
                System.Console.WriteLine($"Lost_iter({k})={Lost(X,Y)}");
                k++;
            }

        }

        private static List<double> Extract_Target(List<double> Y,int start,int batch_size){
            List<double> T=[];
            for (int i = start; i < start+batch_size; i++)
            {
                T.Add(Y[i]);
            }
            return T;
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

        public List<double> Gradient(List<List<double>> x,List<double> y)
        {
            List<double> grad =Zeros(beta!.Count);
            for (int i = 0; i < x.Count; i++)
            {
                double coef = (-2.0f) * (y[i] - Helper(x[i]));
                grad[0]+=coef/x.Count;
                for (int j = 0; j < x[i].Count; j++)
                {
                    grad[j + 1] += (coef * x[i][j]) / x.Count;
                }
            }
            return grad;
        }

        private static List<double> Zeros(int n){
            List<double> zeros=[];
            for (int i = 0; i < n; i++)
            {
                zeros.Add(0);
            }
            return zeros;
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
