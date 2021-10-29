package com.gbhat.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.datasources.v2.jdbc.JDBCTableCatalog;
import org.h2.tools.DeleteDbFiles;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class SparkPushDown {
    private static final String DB_URL = "jdbc:h2:./citydb;user=user1;password=user123";

    private static void createDatabase() throws SQLException, ClassNotFoundException {
        Class.forName("org.h2.Driver");
        Connection conn = DriverManager.getConnection(DB_URL, "user1", "user123");
        conn.prepareStatement("CREATE SCHEMA \"citydb\"").executeUpdate();
        conn.prepareStatement("CREATE TABLE \"citydb\".\"city\" (name TEXT(32) NOT NULL, " +
                "country TEXT(32) NOT NULL, population BIGINT NOT NULL)")
                .executeUpdate();

        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('New York City','USA',8175133)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Shenzhen','China',10358381)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Kolkata','India',4631392)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Houston','USA',2296224)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Delhi','India',10927986)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Brooklyn','USA',2300664)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Tianjin','China',11090314)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Los Angeles','USA',3971883)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES ('Mumbai', 'India', 12691836)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Shanghai','China',22315474)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Bengaluru','India',5104047)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Beijing','China',11716620)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Chicago','USA',2720546)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Guangzhou','China',11071424)").executeUpdate();
        conn.prepareStatement("INSERT INTO \"citydb\".\"city\" VALUES('Chennai','India',4328063)").executeUpdate();
    }


    private static void deleteDatabase() {
        DeleteDbFiles.execute("./", "citydb", true);
    }

    private static SparkSession createSparkSession() {
        SparkConf conf = new SparkConf()
                .set("spark.sql.catalog.h2", JDBCTableCatalog.class.getName())
                .set("spark.sql.catalog.h2.url", DB_URL)
                .set("spark.sql.catalog.h2.driver", "org.h2.Driver")
                .set("spark.sql.catalog.h2.pushDownAggregate", "true")
                .set("spark.sql.catalog.h2.pushDownLimit", "true");

        SparkSession session = SparkSession.builder()
                .appName("Spark Pushdown Optimization")
                .config(conf)
                .master("local[4, 4]").getOrCreate();

        session.sparkContext().setLogLevel("ERROR");    //Suppress all Spark logs except errors
        return session;
    }
    private static void printLine() {
        for(int i = 0; i < 50; i ++)
            System.out.print("---");
        System.out.println();
    }

    public static void main(String[] args) {

        try {
            createDatabase();

            SparkSession session =  createSparkSession();

            printLine();

            System.out.println("No optimization:");

            Dataset<Row> cityDs = session.read().table("h2.citydb.city");
            cityDs.explain();
            cityDs.show();

            printLine();

            System.out.println("Projection pushdown:");

            // Projection pushdown
            Dataset<Row> ds1 = session.sql("select * from h2.citydb.city");
            ds1 = ds1.select("name", "population");
            ds1.explain();
            ds1.show();

            printLine();

            System.out.println("Filter pushdown:");

            // Pushed filters
            Dataset<Row> ds2 = session.sql("select * from h2.citydb.city where population >= 5000000");
            ds2.explain(true);
            ds2.show();

            printLine();

            System.out.println("Filter, Aggregate and group by pushdown:");

            // Pushed filters, group by and aggregates
            Dataset<Row> ds3 = session.sql("select country, MAX(population), " +
                    "MIN(population) from h2.citydb.city where country in ('China', 'India') group by country");
            ds3.explain(true);
            ds3.show();

        } catch (Exception e) {
            System.out.println("Exception: " + e.getMessage());
            e.printStackTrace();
        } finally {
            deleteDatabase();
        }
    }
}
