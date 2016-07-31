package calcSVM;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class ClNonlinearSMO {
	double alpha[] ;	// ラグランジュ定数
	double w[] ;		// 重みベクトル
	int dim;			// 入力データの次元
	double point[];  	// 訓練データ
    int target[] ;		// 訓練データのラベル
    double E[]; 		// エラーキャッシュ

    double b ; // 閾値
    double eps = 0.5;
    double tol = 0.5;
    double C = 1.0;
    int nData;
    double delta = 0.5; 
    
  /******* コンストラクタ ************************/
    public ClNonlinearSMO(){
    	//空
    }
    
  /******* ある点におけるSVM出力 ******************/
    double SVMoutput(int atPoint){
    	double sum = 0.0;
    	for(int k = 0; k < nData ; ++k){
    		sum += alpha[k] * target[k] * kernel(k, atPoint)  ;
    	}
    	return sum+b ;
    }
    
  /******* カーネルの計算 *********************/
    double kernel(int n1, int n2){
        double buff = 0.0;
        for(int t = 0 ; t < dim ; ++t){
            buff += Math.pow(point[dim*n1+t]-point[dim*n2+t] , 2) ;
        }
        return Math.exp(-buff / (2.0 * delta * delta));
    }
    
  /******* a2=H,Lの目的関数を計算する *************/
    double a2_objFunc(int i2, double aHL){
    	double alpBuff = 0.0;
    	double regBuff = 0.0;

    	for(int m = 0 ; m < nData ; ++m){
    		for(int n = 0 ; n < nData ; ++n){
    		   //　α2の更新を反映する
    			if(n == i2)
    			{
    				alpBuff += aHL;
    				if(m == i2){
    					regBuff += aHL * aHL * target[n] * target[m] * kernel(m,n);
    				}else{
    					regBuff += aHL * alpha[n] * target[m] * target[n] * kernel(m,n);
    				}
    			}else{
    				alpBuff += alpha[m];
    				if(m == i2){
    					regBuff += alpha[n] * aHL * target[m] * target[n] * kernel(m,n);
    				}else{
    					regBuff += alpha[m] * alpha[n] * target[m] * target[n] * kernel(m,n);
    				}
    			}
    		}
    	}
    	return alpBuff + regBuff;
    }
    
  /******* takeStep；αの値を更新する *************/
    int takeStep(int i1, int i2, double E2){
        if(i1 == i2) return 0;
        double E1 = SVMoutput(i1) - target[i1];
        int s	  = target[i1] * target[i2];
        double L, H;

     // ボックス制約の上下限の計算
        if(target[i1]!=target[i2]){
            L = Math.max(0.0, alpha[i2] - alpha[i1]);
            H = Math.min(C, C + alpha[i2] - alpha[i1]);
        } else{
            L = Math.max(0.0, alpha[i1] + alpha[i2] - C);
            H = Math.min(C, alpha[i1] + alpha[i2]);
        }

        if(L == H) return 0;

     // カーネルの計算
        double k11 = kernel(i1, i1);
        double k12 = kernel(i1, i2);
        double k22 = kernel(i2, i2);

        double eta = 2 * k12 - k11 - k22;

        double a1, a2;

     // α2の解析解を計算してα1を導出する
        if(eta < 0){
            a2 = alpha[i2] - target[i2] * (E1 - E2) / eta;
            if(a2 < L) a2 = L;
            else if(a2 > H) a2 = H;
        }else{
        	double Lobj = a2_objFunc(i2,L);
        	double Hobj = a2_objFunc(i2,H);
        	if(Lobj > Hobj + eps) a2 = L;
        	else if(Lobj < Hobj - eps) a2 = H;
        	else a2 = alpha[i2];
        }
        if(Math.abs(a2 - alpha[i2]) < eps * (a2 + alpha[i2] + eps)) return 0;
        a1 = alpha[i1] + s * (alpha[i2] - a2);

     // 新たなLagrange乗数を反映して，「しきい値bを更新する
/*        double b_old =b;//エラーキャッシュの更新の時に使う
    
        double b1 = E1 + y1*(a1-alpha[i1])*k11 + y2*(a2-alpha[i2])*k12 + b;
        double b2 = E2 + y1*(a1-alpha[i1])*k12 + y2*(a2-alpha[i2])*k22 + b;
        this.b = (b1+b2)/2;*/

     // エラーキャッシュの更新
    	for(int k = 0; k < nData ; ++k){
    		E[k] +=  target[i1]*(a1-alpha[i1])*kernel(k,i1)+target[i2]*(a2-alpha[i2])*kernel(k,i2);
//    				+target[i2]*(a2-alpha[i2])*kernel(k,i2)+b_old-b;
    	}
     // αの更新
        alpha[i1] = a1;
        alpha[i2] = a2;

        return 1;
    }
    
  /******* examineExample ********************/
    int examineExample(int i2){
    	double y2 = target[i2];//
		int i1 = 0;
		//
		double E2 = SVMoutput(i2) -y2 ;
		double r2 = E2 * y2;

		if ((r2 < -tol && alpha[i2] < C) || (r2 > tol && alpha[i2] > 0)){
			// ヒューリスティックな選択
			int flagEE = 0;
			// 0 < α < C が一つでもあるかどうかを探す
			for(int i = 0; i < nData; ++i){
				if(alpha[i] > 0 && alpha[i] < C) {
					flagEE = 1;
					break;
				}
			}
			// 0 < α < C があればエラーキャッシュの差が最大になるものを探す
			if(flagEE == 1){
				double valMax = 0.0;
				double error = 0.0;
				for(int j = 0; j < nData; ++j){
					error = Math.abs(E[j] - E2);
					if(error > valMax){
                        valMax = error;
                        i1 = j;
					}
					if(takeStep(i1, i2, E2) == 1)return 1;
				}
			}
			// 0 < α < C の中でランダムで選ぶ
			for(int i = 0; i < nData; ++i){
				i1 = (int)((i + Math.random()) % nData);
				// →メルセンヌ・ツイスタを使う必要はないので，Math.randomにした
				if(alpha[i1] > 0 && alpha[i1] < C){
					if(takeStep(i1, i2, E2) == 1)return 1;
				}
			}
			for(int i = 0; i < nData; ++i){
				i1 = (int)((i + Math.random()) % nData);
				// →メルセンヌ・ツイスタを使う必要はないので，Math.randomにした
				if(takeStep(i1, i2, E2) == 1)return 1;
			}
		}
		return 0;
    }
    
  /******* main **************************/
    public void mainSMO(double traiSet[],int traiLabel[],int numData,int dimension){
    	this.point = Arrays.copyOf(traiSet, traiSet.length);
    	this.target = Arrays.copyOf(traiLabel, traiLabel.length);
    	this.nData = numData;
    	this.dim= dimension;
    	this.alpha = new double[nData];
    	this.w = new double[dim];
    	this.E = new double[nData];
    	this.b = 0.0;

        int numChanged = 0;	// αを変更した回数を収集
        int examineAll = 1;	// 全数検査のフラグ
        
      // 計算開始　→　一回のループにおけるαの変更数が0，で全数検査フラグが0ならば終了
        while(numChanged > 0 || examineAll == 1){
            numChanged = 0;
           // 全数検査のフラグが1ならば全部調べる
            if(examineAll==1){
                for(int i=0; i<nData; ++i) numChanged += examineExample(i);
            }
            else{
           // 全数検査のフラグが0で　0 < α < C　ならば調べる
                for(int i=0; i<nData; ++i){
                    if(alpha[i] > 0 && alpha[i] < C) numChanged += examineExample(i);
                }
            }
           // フラグの更新
            if(examineAll == 1) examineAll = 0;
            else if(numChanged == 0) examineAll = 1;
        }
        
      // bの更新       	
        double bBuff = 0.0;
        int countAl = 0;
        
        for(int i=0 ;i<nData ;++i){
        	if(alpha[i] > 0.0){
        		for(int j=0 ;j<nData ;++j){
        			bBuff -= alpha[j]*target[j]*kernel(i,j);
        		}
        		countAl++;
        	}
        }
        this.b = 1.0/countAl*bBuff;
             	
      // ファイルへ書き込み   	
        File File = new File("Lagrange_multiplier"+".txt");
        try{
		    BufferedWriter BW = new BufferedWriter(new FileWriter(File));
		    BW.write("Lagrange_multiplier");
		    BW.newLine();
		    for(int j=0 ;j<numData ;++j){
		    	BW.write(String.valueOf(alpha[j]));
		    	BW.newLine();
		    }
		    BW.write("Threshold "+b);
		    
		    BW.close();
		    }catch(IOException e){
		      System.out.println(e);
		    }
    }
    
    
 //==========　識別 ==========================================
  /******* カーネルの計算 *************************/
    double kernelCF(int n1, int n2){
        double buff = 0.0;
        for(int t = 0 ; t < dim ; ++t){
            buff += Math.pow(point[dim*n1 + t] - data[dim*n2 + t] , 2);
        }
        return Math.exp(-buff / (2.0 * delta * delta));
    }
    
  /******* a識別 classification ****************************/
  	double data[];  	// 訓練データ
    int label[] ;		// 訓練データのラベル
    public void mainCF(double testSet[],int testLabel[],int numTestData,int dimension){
    	this.data = Arrays.copyOf(testSet, testSet.length);
    	this.label = Arrays.copyOf(testLabel, testLabel.length);
    	int decision[] = new int[numTestData];
    	double rate = 0;	// 正解率格納
    	double sum;			// SVM出力のバッファ
    // 識別
      	for(int i = 0; i < numTestData; ++i){
        	sum = b;
        	for(int j=0; j<nData ; ++j){
        		sum += alpha[j] *target[j]* kernelCF(j, i);
        	}
        	if(sum > 0){
        		decision[i] = 1;
        		if(decision[i]==testLabel[i])rate +=1.0;
        	}
        	else{
        		decision[i] = -1;
        		if(decision[i]==testLabel[i])rate +=1.0;
        	}
        }
      	
     // 正解率の判定
      	rate = rate/(double)numTestData;
      	System.out.println("accuracy rate "+rate);
      	
     // ファイルへ書き込み
      	File File = new File("decision"+".txt");
        try{
		    BufferedWriter BW = new BufferedWriter(new FileWriter(File));

		    for(int j=0 ;j<numTestData ;++j){
		    	BW.write(String.valueOf(decision[j]));
		    	BW.newLine();
		    }

		    BW.close();
		    }catch(IOException e){
		      System.out.println(e);
		    }
        }
}
