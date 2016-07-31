import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import calcSVM.ClNonlinearSMO;

public class SVM_main {
	public static void main( String[] args ){

	//　データの取り込み
		int numData = 600;	// データの行数
		int dim = 2;		// データの次元
		int k = 3;			// クロスバリデーションの回数

		String line;
		String[] buff;		// 読み込んだデータを分解するときのバッファ
		double[] dataSet = new double[numData*(dim+2)] ;	// テキストから読み込んだデータを格納する配列
											// +2は「ラベル」と「クロスバリデーション用のデータを分割」するためのフラグ
		double[] traiSet = new double[(numData-numData/k)*dim] ;	// トレーニングデータを格納する配列
		double[] testSet = new double[(numData/k)*dim] ;	// テストデータを格納する配列
		int[] traiLabel = new int[numData-numData/k] ;			// トレーニングデータのラベルを格納する配列
		int[] testLabel = new int[numData/k] ;			// テストデータのラベルを格納する配列
		int count = 0 ;		// カウンタ
		File inputFile = new File("data.csv");
		BufferedReader in = null;

	//　トレーニングデータを配列に格納する
		try {
			in = new BufferedReader(new FileReader(inputFile));

			while ((line = in.readLine()) != null)
			{
				buff = line.split("\\,");

				// データの格納
				for(int i=0 ;i<dim+1 ;++i)
				{
					dataSet[count*(dim+2)+i] = Double.parseDouble( buff[i] );
				}
				count += 1;
			}
		}catch(FileNotFoundException e){
			e.printStackTrace();
		}catch(IOException e){
			e.printStackTrace();
		}finally{
			try{
				if (in != null){
					in.close();
				}

			}catch(IOException e){
				System.out.println("close");
				e.printStackTrace();
			}
		}
	//　データの中身を確認する
//		for(int i=0 ;i<numData*(dim+2) ;++i)System.out.println(dataSet[i]);

	//　データを分ける
		ClNonlinearSMO SVMc = new ClNonlinearSMO();
		int countP ;
		int countM ;
		int times  ;

		for(int i=0 ;i<k ;++i){
			times = i+1;
			File traiFile = new File("traiSet_iteration"+times+".txt");
			File testFile = new File("testSet_iteration"+times+".txt");
			countP = 0;
			countM = 0;
			count  = 0 ;
		// テストデータの生成
			for(int j=0 ;j<numData ;++j){
				// 「ラベルが+1」かつ「フラグがk番目（プログラム上ではｋ+1番目）が立っていない」ならばテストデータとして採用する
				if(dataSet[j*(dim+2)+dim]==1.0 && dataSet[j*(dim+2)+dim+1] ==0 && countP < numData/(2*k)){
					for(int n=0 ;n<dim ;++n){
						testSet[dim*countP+n] = dataSet[j*(dim+2)+n];
					}
					testLabel[countP] = (int)(dataSet[j*(dim+2)+dim]);
					dataSet[j*(dim+2)+dim+1] = i+1;
					countP += 1;
//					System.out.println(countP);//+++
				}
				// 「ラベルが-1」かつ「フラグがk番目（プログラム上ではｋ+1番目）が立っていない」ならばテストデータとして採用する
				if(dataSet[j*(dim+2)+dim]== -1.0 && dataSet[j*(dim+2)+dim+1] ==0 && countM < numData/(2*k)){
					for(int n=0 ;n<dim ;++n){
						testSet[numData/(2*k)*dim+countM*dim+n] = dataSet[j*(dim+2)+n];
					}
					testLabel[numData/(2*k)+countM] = (int)(dataSet[j*(dim+2)+dim]);
					dataSet[j*(dim+2)+dim+1] = i+1;
					countM += 1;
				}
				count += 1;
			}

		// トレーニングデータの生成
			countP = 0;
			countM = 0;
			count  = 0;
			for(int j=0 ;j<numData ;++j){
				if(dataSet[j*(dim+2)+dim] == 1.0 && dataSet[j*(dim+2)+dim+1] < i+1 && countP<(numData-numData/k)/2){
					for(int n=0 ;n<dim ;++n){
						traiSet[countP*dim+n] = dataSet[j*(dim+2)+n];
					}
					traiLabel[countP] = (int)(dataSet[j*(dim+2)+dim]);
					countP += 1;
				}
				if(dataSet[j*(dim+2)+dim] == -1.0 && dataSet[j*(dim+2)+dim+1] < i+1 && countM<(numData-numData/k)/2){
					for(int n=0 ;n<dim ;++n){
						traiSet[(numData-numData/k)/2*dim+countM*dim+n] = dataSet[j*(dim+2)+n];
					}
					traiLabel[(numData-numData/k)/2+countM] = (int)(dataSet[j*(dim+2)+dim]);
					countM += 1;
				}
			}
		//　データの中身を確認する
			try{
			    BufferedWriter traiBW = new BufferedWriter(new FileWriter(traiFile));
			    BufferedWriter testBW = new BufferedWriter(new FileWriter(testFile));

			    traiBW.write("iteration"+times);traiBW.newLine();
			    testBW.write("iteration"+times);testBW.newLine();

			    for(int j=0 ;j<(numData-numData/k) ;++j){
			    	for(int n=0 ;n<dim ;++n){
			    		traiBW.write(traiSet[dim*j+n]+",");
			    	}
			    	traiBW.write(String.valueOf(traiLabel[j]));// intなので一度Stringに直さないと書き込めないので注意！
			    	traiBW.newLine();
			    }
			    for(int j=0 ;j<numData/k ;++j){
			    	for(int n=0 ;n<dim ;++n){
			    		testBW.write(testSet[dim*j+n]+",");
			    	}
			    	testBW.write(String.valueOf(testLabel[j]));// intなので一度Stringに直さないと書き込めないので注意！
			    	testBW.newLine();
			    }

			    traiBW.close();
			    testBW.close();

			    }catch(IOException e){
			      System.out.println(e);
			  }
			/***** トレーニング *****/
			SVMc.mainSMO(traiSet,traiLabel,numData-numData/k,dim);
			SVMc.mainCF(testSet,testLabel,numData/k,dim);
		}


	}
}
