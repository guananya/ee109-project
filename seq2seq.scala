import spatial.dsl._
import virtualized._

@spatial class Seq2SeqInference extends SpatialTest {

  val vocabSize = 56
  val maxTextLength = 52
  val flattenedSize = vocabSize * maxTextLength
  val dense1Size = 512
  type T = FixPt[TRUE,_16,_16] 
  
  def main(args: Array[String]): Unit = {
    // Load data
    val inputDRAM = DRAM[T](flattenedSize)
    val weight0DRAM = DRAM[T](flattenedSize, dense1Size)
    val weight1DRAM = DRAM[T](dense1Size)
    val weight2DRAM = DRAM[T](dense1Size, vocabSize)
    val weight3DRAM = DRAM[T](vocabSize)
    val outputDRAM = DRAM[T](maxTextLength, vocabSize)

    val weight0 = loadCSV2D[T]("weight_0.csv")
    val weight1 = loadCSV1D[T]("weight_1.csv")
    val weight2 = loadCSV2D[T]("weight_2.csv")
    val weight3 = loadCSV1D[T]("weight_3.csv")

    val inputArray = loadCSV1D[T]("input_one_hot_flattened.csv")

    setMem(inputDRAM, inputArray)
    setMem(weight0DRAM, weight0)
    setMem(weight1DRAM, weight1)
    setMem(weight2DRAM, weight2)
    setMem(weight3DRAM, weight3)

    Accel {
      val w0 = SRAM[T](flattenedSize, dense1Size)
      val w1 = SRAM[T](dense1Size)
      val w2 = SRAM[T](dense1Size, vocabSize)
      val w3 = SRAM[T](vocabSize)

      w0 load weight0DRAM
      w1 load weight1DRAM
      w2 load weight2DRAM
      w3 load weight3DRAM

      val inData = SRAM[T](flattenedSize)
      inData load inputDRAM

      val dense1 = SRAM[T](dense1Size)
      val output = SRAM[T](maxTextLength, vocabSize)

      Foreach(dense1Size by 1) { i =>
        dense1(i) = max(0.to[T], inData dot w0(0::flattenedSize, i) + w1(i))
      }

      Foreach(maxTextLength by 1) { i =>
        val sumExp = Reg[T]
        val preSoftmax = SRAM[T](vocabSize)

        Reduce(sumExp)(vocabSize by 1) { j =>
          preSoftmax(j) = exp(dense1 dot w2(0::dense1Size, j) + w3(j))
          preSoftmax(j)
        } { _ + _ }

        Foreach(vocabSize by 1) { j =>
          output(i, j) = preSoftmax(j) / sumExp.value
        }
      }

      outputDRAM(0::maxTextLength, 0::vocabSize) store output
    }

    val result = getMatrix(outputDRAM)
    printMatrix(result, "Output")
  }
}
