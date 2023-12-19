//+------------------------------------------------------------------+
//|                                            Linear Regression.mqh |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan" 

//+------------------------------------------------------------------+

#include <MALE5\metrics.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\preprocessing.mqh>

//+------------------------------------------------------------------+

class CLinearRegression
  {
   private:
   
   CMetrics metrics;
   CMatrixutils matrix_utils;
   CPreprocessing *normalize_x;
  
   protected:
                        bool istrained;          
                        bool checkIsTrained(string func)
                          {
                            if (!istrained)
                              {
                                Print(func," Tree not trained, Call fit function first to train the model");
                                return false;   
                              }
                            return (true);
                          }
                           
                        template<typename T>
                        T dx_wrt_bo(matrix<T> &x, vector<T> &y_vector);
                        template<typename T>
                        vector<T> dx_wrt_b1(matrix<T> &x, vector<T> &y_vector);
                        
       
   public:
                        matrix Betas;   //Coefficients matrix
                        vector Betas_v; //Coefficients vector 
                        
                        //double residual_value;  //Mean residual value
                        //vector Residuals;
                        
                        CLinearRegression(void);
                       ~CLinearRegression(void);
                       
                        template <typename T>
                        void fit(matrix<T> &x, vector<T> &y, norm_technique NORM_METHOD); //Least squares estimator
                        template <typename T>
                        void fit(matrix<T> &x, vector<T> &y, norm_technique NORM_METHOD, double alpha, uint iters = 1000); //LR by Gradient descent
                        
                        template <typename T>
                        T pred(vector<T> &x); 
                        template <typename T>
                        vector<T> pred(matrix<T> &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinearRegression::CLinearRegression(void) : istrained(false)
 {
       
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
void CLinearRegression::fit(matrix<T> &x, vector<T> &y, norm_technique NORM_METHOD)
 {
  matrix temp_x =x; 
 
  matrix YMatrix = {};
  YMatrix = matrix_utils.VectorToMatrix(y);
  
  normalize_x = new CPreprocessing(temp_x, NORM_METHOD); 
    
    ulong rows = y.Size(); 
    ulong cols = temp_x.Cols();
    
    if (rows != temp_x.Rows())
      {
         Print(__FUNCTION__," FATAL: Unbalanced rows ",rows," in the independent vector and x matrix of ",x.Rows()," rows");
         return;
      }
      
//---

    matrix design = matrix_utils.DesignMatrix(temp_x);
    
//--- XTX
    
    matrix XT = design.Transpose();
    
    matrix XTX = XT.MatMul(design);
    
//--- Inverse XTX

    matrix InverseXTX = XTX.Inv();
    
//--- Finding XTY
   
    matrix XTY = XT.MatMul(YMatrix);

//--- Coefficients
   
   Betas = InverseXTX.MatMul(XTY); 
   
   Betas_v = matrix_utils.MatrixToVector(Betas);
   
   #ifdef DEBUG_MODE 
        Print("Betas\n",Betas);
   #endif 
   
   istrained = true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
void CLinearRegression::fit(matrix<T> &x, vector<T> &y, norm_technique NORM_METHOD, double alpha, uint iters = 1000)
 {     
   matrix temp_x = x;
   matrix YMatrix = matrix_utils.VectorToMatrix(y);
   
   normalize_x = new CPreprocessing(temp_x, NORM_METHOD)
       
   y = matrix_utils.MatrixToVector(YMatrix); 
  
    ulong rows = y.Size();
    ulong cols = temp_x.Cols();
    
    if (rows != temp_x.Rows())
      {
         Print("FATAL: Unbalanced rows in the independent vector and x matrix");
         return;
      }
//---

    iterations = iters;
    
    Betas_v.Resize(cols+1);

//---
     #ifdef DEBUG_MODE  
        Print("\nTraining a Linear Regression Model with Gradient Descent\n");
     #endif 
//---
     
     for (ulong i=0; i<iterations; i++)
       {
       
         if (i==0) Betas_v.Fill(0);

//---

         double bo = dx_wrt_bo(temp_x,y);
         
         Betas_v[0] = Betas_v[0] - (alpha * bo);
         //printf("----> dx_wrt_bo | Intercept = %.8f | Real Intercept = %.8f",bo,Betas_v[0]);
         
         vector dx = dx_wrt_b1(temp_x,y); 

//---

          for (ulong j=0; j<dx.Size(); j++)
            {
               //Print("out at iterations Betas _v ",Betas_v);
                
                  Betas_v[j+1] = Betas_v[j+1] - (alpha * dx[j]);
                  
                  //printf("k %d | ----> dx_wrt_b%d | Slope = %.8f | Real Slope = %.8f",j,j,dx[j],Betas_v[j+1]); 
            }
         
//---

           #ifdef DEBUG_MODE  
               Betas = matrix_utils.VectorToMatrix(Betas_v);
               double acc =0;
               //Print("Betas ",Betas);
                
               vector pred = pred(temp_x);
                matrix_utils.NormalizeVector(dx,5);
               
               Print("[ ",i+1," ] Accuracy = ",NormalizeDouble(metrics.r_squared(y,pred)*100,2),"% | COST ---> WRT Intercept | ",NormalizeDouble(bo,5)," | WRT Coeff ",dx);

           #endif  
           
       } 
//---
    Betas = matrix_utils.VectorToMatrix(Betas_v);
//---

    #ifdef DEBUG_MODE 
        matrix_utils.NormalizeVector(Betas_v,5);
        Print("Coefficients ",Betas_v);
    #endif 
    
    istrained = true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinearRegression::~CLinearRegression(void)
 {   
   delete (normalize_x);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
T CLinearRegression::dx_wrt_bo(matrix<T> &x, vector<T> &y_vector)
 {    
   T mx=0, sum=0;
   for (ulong i=0; i<x.Rows(); i++)
      {          
          mx = pred(x.Row(i));
          
          sum += (y_vector[i] - mx);  
      }  
   
   return(-2*sum);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector<T> CLinearRegression::dx_wrt_b1(matrix<T> &x, vector<T> &y_vector)
 { 
   vector<T> dx_vector(Betas_v.Size()-1);
   //Print("dx_vector.Size() = ",dx_vector.Size());
   
    double mx=0, sum=0;
   
    for (ulong b=0; b<dx_vector.Size(); b++)  
     {
       ZeroMemory(sum);
       
       for (ulong i=0; i<x.Rows(); i++)
         {             
             //Print("<<<    >>> intercept = ",mx," Betas_v ",Betas_v,"\n");
             
            mx = pred(x.Row(i));            

//---

            sum += (y_vector[i] - mx) * x[i][b];  
            //PrintFormat("%d xMatrix %.5f",i,x[i][b]); 
          
            dx_vector[b] = -2*sum;  
        }
    }
      
    return dx_vector;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
T CLinearRegression::pred(vector<T> &x)
 {
   vector <T> temp_x = x;
   normalize_x.Normalization(temp_x);
   
   if (!checkIsTrained(__FUNCTION__))
     return 0;
   
   double pred_value =0; 
   double intercept = Betas_v[0];
   
   if (Betas_v.Size() == 0)
      {
         Print(__FUNCTION__,"Err, No coefficients available for LR model\nTrain the model before attempting to use it");
         return(0);
      }
   
    else
      { 
        if (temp_x.Size() != Betas_v.Size()-1)
          Print(__FUNCTION__,"Err, X vars not same size as their coefficients vector ");
        else
          {
            for (ulong i=1; i<Betas_v.Size(); i++) 
               pred_value += temp_x[i-1] * Betas_v[i];  
               
            pred_value += intercept; // + residual_value; 
          }
      }
    return  pred_value;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
vector<T> CLinearRegression::pred(matrix<T> &x)
 {
   vector<T> pred_v(x.Rows());
   vector<T> x_vec;
   
    for (ulong i=0; i<x.Rows(); i++)
     {
       x_vec = x.Row(i);
       pred_v[i] = pred(x_vec);
     }
   
   return pred_v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
