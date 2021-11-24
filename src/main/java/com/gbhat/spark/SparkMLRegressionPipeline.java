package com.gbhat.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;

public class SparkMLRegressionPipeline {
    public static void main(String[] args) {
        SparkSession spark = buildSparkSession();

        Dataset<Row> inDs = read(spark);

        inDs.printSchema();

        inDs.select("Origin").distinct().show();

        inDs = inDs.drop("Car Name");

        inDs.summary().show();

        Dataset<Row> transformedDs = transform(inDs);

        transformedDs.select("Origin", "OriginVec")
                .withColumn("OriginVecDense", functions.udf((UDF1<Vector, Object>) Vector::toDense, new VectorUDT())
                        .apply(transformedDs.col("OriginVec"))).distinct().show();

        Dataset<Row>[] splits = transformedDs.randomSplit(new double[] {0.7, 0.3}, 0);
        Dataset<Row> trainDs = splits[0];
        Dataset<Row> testDs = splits[1];

        linearRegression(trainDs, testDs);

        randomForestRegression(trainDs, testDs);
    }

    private static void randomForestRegression(Dataset<Row> trainDs, Dataset<Row> testDs) {
        RandomForestRegressor rfRegressor = new RandomForestRegressor()
                .setNumTrees(8)
                .setSeed(0)
                .setFeaturesCol("TrainFeaturesScaled")
                .setLabelCol("MPG");
        RandomForestRegressionModel rfModel = rfRegressor.fit(trainDs);

        Dataset<Row> predictions = rfModel.transform(testDs);
        predictions = predictions.select("MPG", "prediction");
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("MPG")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Random Forest Regression RMSE on test data = " + rmse);
    }

    private static void linearRegression(Dataset<Row> trainDs, Dataset<Row> testDs) {
        LinearRegression lr = new LinearRegression()
                .setRegParam(0.3)
                .setFeaturesCol("TrainFeaturesScaled")
                .setLabelCol("MPG");
        LinearRegressionModel lrModel = lr.fit(trainDs);

        Dataset<Row> predictions = lrModel.transform(testDs);
        predictions = predictions.select("MPG", "prediction");
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("MPG")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Linear Regression RMSE on test data = " + rmse);
    }

    private static Dataset<Row> transform(Dataset<Row> inDs) {
        Imputer imputer = new Imputer()
		.setStrategy("median")
                .setMissingValue(0)
                .setInputCols(new String[]{"Horsepower"})
                .setOutputCols(new String[]{"Horsepower"});

        StringIndexer strIndexer = new StringIndexer()
                .setInputCol("Origin")
                .setOutputCol("OriginIdx");

        OneHotEncoder oneHotencoder = new OneHotEncoder()
                .setInputCol("OriginIdx")
                .setOutputCol("OriginVec");

        VectorAssembler vecAssembler = new VectorAssembler()
                .setInputCols(new String[]{"Cylinders", "Displacement",
                        "Horsepower", "Weight", "Acceleration", "Model Year", "OriginVec"})
                .setOutputCol("TrainFeatures");

        StandardScaler stdScaler = new StandardScaler()
                .setInputCol("TrainFeatures")
                .setOutputCol("TrainFeaturesScaled");

        Pipeline pipeLine = new Pipeline()
                .setStages(new PipelineStage[]{imputer, strIndexer, oneHotencoder, vecAssembler, stdScaler});

        Dataset<Row> transformedDs = pipeLine.fit(inDs).transform(inDs);

        return transformedDs;
    }

    private static Dataset<Row> read(SparkSession spark) {
        return spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .option("delimiter", "|")
                .csv("auto-mpg.csv.gz");
    }

    private static SparkSession buildSparkSession() {
        SparkSession session = SparkSession
                .builder()
                .master("local[4, 4]")
                .appName("Spark ML Regression Pipeline")
                .getOrCreate();
        session.sparkContext().setLogLevel("WARN");
        return session;
    }
}
