package main;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class TestWeka {

	public static void main(String[] args) {

		try {
			// データの読み込み
			Instances trainData = new Instances(new BufferedReader(new FileReader("./src/data/iris.arff"))); // ARFF形式で保存したデータの読み込み
			trainData.setClassIndex(trainData.numAttributes() - 1); // 一番最後にTrue/Falseデータを記載しているので「-1」する。

			// 主成分分析
			InfoGainAttributeEval infoGain = new InfoGainAttributeEval();
			AttributeSelection attributeSelection = new AttributeSelection();
			Ranker search = new Ranker(); // SearchとしてRankerを選択

			attributeSelection.setEvaluator(infoGain);
			attributeSelection.setSearch(search);
			attributeSelection.SelectAttributes(trainData);

			attributeSelection.setFolds(10);
			attributeSelection.setSeed(1);
			attributeSelection.selectAttributesCVSplit(trainData);
			System.out.println(attributeSelection.CVResultsString());
			System.out.println(attributeSelection.toResultsString());

			// ナイーブベイズ処理
			weka.classifiers.bayes.NaiveBayes scheme = new weka.classifiers.bayes.NaiveBayes();

			scheme.setOptions(weka.core.Utils.splitOptions(""));
			scheme.buildClassifier(trainData); // NaiveBayesを実行

			// 10fold CV with seed=1 での交差検定
			Evaluation m_Evaluation = new Evaluation(trainData);
			m_Evaluation.crossValidateModel(scheme, trainData, 10, trainData.getRandomNumberGenerator(1));
			System.out.println("pre 0:" + m_Evaluation.precision(0) + "\n");
			System.out.println("pre 1:" + m_Evaluation.precision(1) + "\n");

			System.out.println(m_Evaluation.toSummaryString() + "\n");
		} catch (Exception e) {
			System.out.println("ERROR");
		}
	}
}
