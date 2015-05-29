import AssemblyKeys._

assemblySettings

name := "template-scala-parallel-vanilla"

organization := "io.prediction"

resolvers += Resolver.mavenLocal

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "io.prediction" %% "core" % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core" % "1.3.0" % "provided",
  "org.xerial.snappy" % "snappy-java" % "1.1.1.7",
  "com.github.tototoshi" %% "scala-csv" % "1.2.1",
  "edu.stanford.nlp" % "stanford-parser" % "3.5.2",
  "cml" %% "cml" % "0.1.2-SNAPSHOT"
)
