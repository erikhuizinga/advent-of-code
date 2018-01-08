package io.github.erikhuizinga.adventofcode

import io.github.erikhuizinga.adventofcode.Operation.*
import java.net.URL
import kotlin.math.*

/** Created by erik.huizinga on 27-12-17 */
fun main(args: Array<String>) = day11Part1()

//<editor-fold desc="Day 1">
val captcha1: List<String> = listOf("1122", "1111", "1234", "91212129", "823936645345581272695677318513459491834641129844393742672553544439126314399846773234845535593355348931499496184839582118817689171948635864427852215325421433717458975771369522138766248225963242168658975326354785415252974294317138511141826226866364555761117178764543435899886711426319675443679829181257496966219435831621565519667989898725836639626681645821714861443141893427672384716732765884844772433374798185955741311116365899659833634237938878181367317218635539667357364295754744829595842962773524584225427969467467611641591834876769829719248136613147351298534885563144114336211961674392912181735773851634298227454157885241769156811787611897349965331474217223461176896643242975397227859696554492996937235423272549348349528559432214521551656971136859972232854126262349381254424597348874447736545722261957871275935756764184378994167427983811716675476257858556464755677478725146588747147857375293675711575747132471727933773512571368467386151966568598964631331428869762151853634362356935751298121849281442128796517663482391226174256395515166361514442624944181255952124524815268864131969151433888721213595267927325759562132732586252438456569556992685896517565257787464673718221817783929691626876446423134331749327322367571432532857235214364221471769481667118117729326429556357572421333798517168997863151927281418238491791975399357393494751913155219862399959646993428921878798119215675548847845477994836744929918954159722827194721564121532315459611433157384994543332773796862165243183378464731546787498174844781781139571984272235872866886275879944921329959736315296733981313643956576956851762149275521949177991988236529475373595217665112434727744235789852852765675189342753695377219374791548554786671473733124951946779531847479755363363288448281622183736545494372344785112312749694167483996738384351293899149136857728545977442763489799693492319549773328626918874718387697878235744154491677922317518952687439655962477734559232755624943644966227973617788182213621899579391324399386146423427262874437992579573858589183571854577861459758534348533553925167947139351819511798829977371215856637215221838924612644785498936263849489519896548811254628976642391428413984281758771868781714266261781359762798")

private fun day1Part1() {
  for (captcha in captcha1) {
    println(solveCaptcha1(captcha))
  }
}

fun solveCaptcha1(captcha: String): Int =
    captcha.run {
      plus(first())
          .zipWithNext()
          .filter { it.first == it.second }
          .sumBy { it.first.toString().toInt() }
    }

val captcha2: List<String> = listOf("1212", "1221", "123425", "123123", "12131415", captcha1.last())

private fun day1Part2() {
  for (captcha in captcha2) {
    println(solveCaptcha2(captcha))
  }
}

fun solveCaptcha2(captcha: String): Int =
    captcha.run {
      zip(drop(length / 2))
          .filter { it.first == it.second }
          .sumBy { 2 * it.first.toString().toInt() }
    }
//</editor-fold>

//<editor-fold desc="Day 2">
val inputDay2 = "1136\t1129\t184\t452\t788\t1215\t355\t1109\t224\t1358\t1278\t176\t1302\t186\t128\t1148\n" +
    "242\t53\t252\t62\t40\t55\t265\t283\t38\t157\t259\t226\t322\t48\t324\t299\n" +
    "2330\t448\t268\t2703\t1695\t2010\t3930\t3923\t179\t3607\t217\t3632\t1252\t231\t286\t3689\n" +
    "89\t92\t903\t156\t924\t364\t80\t992\t599\t998\t751\t827\t110\t969\t979\t734\n" +
    "100\t304\t797\t81\t249\t1050\t90\t127\t675\t1038\t154\t715\t79\t1116\t723\t990\n" +
    "1377\t353\t3635\t99\t118\t1030\t3186\t3385\t1921\t2821\t492\t3082\t2295\t139\t125\t2819\n" +
    "3102\t213\t2462\t116\t701\t2985\t265\t165\t248\t680\t3147\t1362\t1026\t1447\t106\t2769\n" +
    "5294\t295\t6266\t3966\t2549\t701\t2581\t6418\t5617\t292\t5835\t209\t2109\t3211\t241\t5753\n" +
    "158\t955\t995\t51\t89\t875\t38\t793\t969\t63\t440\t202\t245\t58\t965\t74\n" +
    "62\t47\t1268\t553\t45\t60\t650\t1247\t1140\t776\t1286\t200\t604\t399\t42\t572\n" +
    "267\t395\t171\t261\t79\t66\t428\t371\t257\t284\t65\t25\t374\t70\t389\t51\n" +
    "3162\t3236\t1598\t4680\t2258\t563\t1389\t3313\t501\t230\t195\t4107\t224\t225\t4242\t4581\n" +
    "807\t918\t51\t1055\t732\t518\t826\t806\t58\t394\t632\t36\t53\t119\t667\t60\n" +
    "839\t253\t1680\t108\t349\t1603\t1724\t172\t140\t167\t181\t38\t1758\t1577\t748\t1011\n" +
    "1165\t1251\t702\t282\t1178\t834\t211\t1298\t382\t1339\t67\t914\t1273\t76\t81\t71\n" +
    "6151\t5857\t4865\t437\t6210\t237\t37\t410\t544\t214\t233\t6532\t2114\t207\t5643\t6852"

fun day2Part1() =
    println(
        inputDay2
            .split("\n")
            .map {
              it
                  .split("\t")
                  .map(String::toInt)
                  .run { max()!! - min()!! }
            }
            .sum()
    )

fun day2Part2() =
    println(
        inputDay2
            .split("\n")
            .map { line ->
              line
                  .split("\t")
                  .map(String::toInt)
                  .run {
                    mapNotNull { numerator ->
                      minus(numerator)
                          .singleOrNull { denominator -> numerator.rem(denominator) == 0 }
                          ?.let { denominator -> numerator / denominator }
                    }
                  }
                  .single()
            }
            .sum()
    )
//</editor-fold>

//<editor-fold desc="Day 3">
val inputDay3 = 277678
val inputsDay3: IntArray = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 23, 1024, inputDay3)

fun day3Part1() {
  inputsDay3.forEach { println(Pair(it, it.spirallingManhattanDistance())) }
}

val maxIntSqrt = truncate(sqrt(Int.MAX_VALUE.toDouble())).toInt()

fun Int.spirallingManhattanDistance(): Int =
    (1..maxIntSqrt)
        .last { root -> (root * root) <= this }
        .let { root ->
          root - 1 +
              min(
                  (this - root * root),
                  min(
                      this - root * root - root / 2 - 1,
                      root * root + 3 / 2 * root + root % 2 + 1 - this
                  ).absoluteValue
                      - root / 2 + 1
              )
        }

typealias Position = Pair<Int, Int>

val Position.row get() = first
val Position.col get() = second

operator fun Position.plus(other: Position): Position = row + other.row to col + other.col
operator fun Position.minus(other: Position): Position = row - other.row to col - other.col

fun Position.nextLeft(previous: Position): Position =
    (this - previous).let {
      when {
        it.row.absoluteValue + it.col.absoluteValue != 1 -> throw error(previous)
        it.row == 1 -> Position(row, col - 1)
        it.row == -1 -> Position(row, col + 1)
        it.col == 1 -> Position(row + 1, col)
        it.col == -1 -> Position(row - 1, col)
        else -> throw error(previous)
      }
    }

private fun Position.error(previous: Position) =
    IllegalArgumentException("Previous position ($previous) must be at exactly one Manhattan " +
        "distance from this position (${this})")

fun Position.nextStraight(previous: Position): Position = this + this - previous

data class SpiralCell(
    private val position: Position = Position(0, 0),
    private val existingCells: List<SpiralCell> = emptyList()
) {

  private val existingNeighbors: Collection<SpiralCell> =
      existingCells.filter {
        (it.position - position).let { it.row.absoluteValue <= 1 && it.col.absoluteValue <= 1 }
      }

  val value: Int = max(1, existingNeighbors.sumBy { it.value })

  fun next() = SpiralCell(nextPosition(), existingCells + this)

  private fun nextPosition(): Position =
      existingCells
          .lastOrNull()
          ?.position
          ?.let { previousPosition ->
            position
                .nextLeft(previousPosition)
                .takeUnless { it in existingNeighbors.map { it.position } }
                ?: position.nextStraight(previousPosition)
          }
          ?: Position(position.row, position.col + 1)

  override fun toString(): String = "$value $position"
}

fun day3Part2() =
    mutableListOf(SpiralCell())
        .run {
          while (last().value <= inputDay3) {
            add(last().next())
          }
          println(last())
        }
//</editor-fold>

//<editor-fold desc="Day 4">
val testInputsDay4Part1 = listOf("aa bb cc dd ee", "aa bb cc dd aa", "aa bb cc dd aaa")

val inputDay4File = FileProvider.url("/inputDay4.txt")

object FileProvider {
  fun url(name: String): URL? = javaClass.getResource(name)
}

val predicateDay4Part1 = String::isValidPassphrase

fun testDay4Part1() =
    testInputsDay4Part1.forEach { println(it.countValidPassphrases(predicateDay4Part1)) }

fun day4Part1() =
    println(inputDay4File?.readText()?.countValidPassphrases(predicateDay4Part1))

private fun String.countValidPassphrases(predicate: (String) -> Boolean): Int =
    split("\n").filter { it.isNotEmpty() }.count { predicate(it) }

private fun String.isValidPassphrase(): Boolean = split(" ").run { size == toSet().size }

val testInputsDay4Part2 = listOf(
    "abcde fghij",
    "abcde xyz ecdab",
    "a ab abc abd abf abj",
    "iiii oiii ooii oooi oooo",
    "oiii ioii iioi iiio"
)

fun testDay4Part2() = testInputsDay4Part2.forEach { println(it.isValidPassphraseAnagram()) }

private fun String.isValidPassphraseAnagram(): Boolean =
    split(" ")
        .map { s -> s.toList().sorted() }
        .run { size == toSet().size }

fun day4Part2() =
    println(inputDay4File?.readText()?.countValidPassphrases(String::isValidPassphraseAnagram))
//</editor-fold>

//<editor-fold desc="Day 5">
val testInputDay5 = intArrayOf(0, 3, 0, 1, -3)

fun testDay5Part1() = println(testInputDay5.countMazeJumps())

private fun IntArray.countMazeJumps(): Int {
  var i = 0
  var steps = 0
  while (i < size) {
    val value = get(i)
    set(i, value + 1)
    i += value
    steps++
  }
  return steps
}

val inputDay5File = FileProvider.url("/inputDay5.txt")

fun day5Part1() = println(
    inputDay5File
        ?.readText()
        ?.split("\n")
        ?.filter(String::isNotEmpty)
        ?.map(String::toInt)
        ?.toIntArray()
        ?.countMazeJumps()
)

fun testDay5Part2() = println(testInputDay5.countMazeJumps2())

private fun IntArray.countMazeJumps2(): Int {
  var i = 0
  var steps = 0
  while (i < size) {
    val value = get(i)
    set(i, value + if (value < 3) 1 else -1)
    i += value
    steps++
  }
  return steps
}

fun day5Part2() = println(
    inputDay5File
        ?.readText()
        ?.split("\n")
        ?.filter(String::isNotEmpty)
        ?.map(String::toInt)
        ?.toIntArray()
        ?.countMazeJumps2()
)
//</editor-fold>

//<editor-fold desc="Day 6">
val testInputDay6 = intArrayOf(0, 2, 7, 0)

fun testDay6Part1() = day6Part1Fun(testInputDay6)

private fun day6Part1Fun(memory: IntArray) {
  val history = mutableSetOf<IntArray>()
  while (history.all { !it.contentEquals(memory) }) {
    history += memory.copyOf()
    memory.balance()
  }

  history.run {
    println("Balancing repetition after $size steps, " +
        "infinite loop size: ${size - indexOf(find { it.contentEquals(memory) })}")
  }
}

private fun IntArray.balance() =
    run {
      val max = max()!!
      val iMax = indexOf(max)
      this[iMax] = 0
      (iMax + 1..max + iMax).forEach { index -> this[index % size]++ }
    }

val inputDay6 = intArrayOf(5, 1, 10, 0, 1, 7, 13, 14, 3, 12, 8, 10, 7, 12, 0, 6)

fun day6Part1And2() = day6Part1Fun(inputDay6)
//</editor-fold>

//<editor-fold desc="Day 7">
val testInputDay7 =
    "pbga (66)\n" +
        "xhth (57)\n" +
        "ebii (61)\n" +
        "havc (66)\n" +
        "ktlj (57)\n" +
        "fwft (72) -> ktlj, cntj, xhth\n" +
        "qoyq (66)\n" +
        "padx (45) -> pbga, havc, qoyq\n" +
        "tknk (41) -> ugml, padx, fwft\n" +
        "jptl (61)\n" +
        "ugml (68) -> gyxo, ebii, jptl\n" +
        "gyxo (61)\n" +
        "cntj (57)"

val inputDay7 = FileProvider.url("/inputDay7.txt")!!.readText()

fun testDay7Part1() = println(findBottomProgramName(testInputDay7))

fun day7Part1() = println(findBottomProgramName(inputDay7))

val nameRegex = "[A-z]+".toRegex()

fun findBottomProgramName(input: String): String =
    input
        .run { nameRegex.findAll(this) }
        .map(MatchResult::value)
        .run { single { programName -> programName !in minus(programName) } }

fun testDay7Part2() = println(findTargetWeight(testInputDay7))

fun day7Part2() = println(findTargetWeight(inputDay7))

fun findTargetWeight(input: String): Int {
  val bottomProgramName = findBottomProgramName(input)
  val weightGroups = generateSequence(
      Program.parse(input).single { it.name == bottomProgramName }.subPrograms
  ) {
    it
        .groupBy(Program::totalWeight)
        .values
        .singleOrNull {
          it.size == 1
              && it.first().subPrograms.map { it.totalWeight }.distinct().size > 1
        }
        ?.first()
        ?.subPrograms
  }
      .last()
      .groupBy(Program::totalWeight)
  val unbalancedProgram = weightGroups.values.single { it.size == 1 }.first()
  return (weightGroups.keys - unbalancedProgram.totalWeight)
      .single() - unbalancedProgram.totalWeight + unbalancedProgram.weight
}

class Program(
    val name: String,
    val weight: Int,
    val subPrograms: MutableSet<Program> = mutableSetOf()
) {

  val totalWeight: Int
    get() = weight + subPrograms.sumBy(Program::totalWeight)

  companion object {
    fun parse(input: String): Set<Program> {
      val programLines = input.split('\n').filterNot(String::isEmpty)
      return programLines
          .mapNotNull {
            Program(
                nameRegex.find(it)?.value ?: return@mapNotNull null,
                "\\d+".toRegex().find(it)?.value?.toInt() ?: return@mapNotNull null
            )
          }
          .toSet() // Set of programs with subprograms not yet added
          .apply {
            // Find and add all subprograms
            programLines.forEach { programLine ->
              nameRegex
                  .findAll(programLine)
                  .map(MatchResult::value)
                  .let { names ->
                    single { program -> program.name == names.first() }
                        .subPrograms
                        .addAll(filter { it.name in names.drop(1) })
                  }
            }
          }
    }
  }
}
//</editor-fold>

//<editor-fold desc="Day 8">
val testInputDay8 =
    "b inc 5 if a > 1\n" +
        "a inc 1 if b < 5\n" +
        "c dec -10 if a >= 1\n" +
        "c inc -20 if c == 10"

val inputDay8 = FileProvider.url("/inputDay8.txt")!!.readText()

fun testDay8Part1And2() = execute(testInputDay8)

fun day8Part1And2() = execute(inputDay8)

private fun execute(input: String) {
  Instruction.parse(input).forEach { it.execute() }
  println("Max after execution = ${Instruction.register.values.max()}")
  println("Max ever during execution = ${Instruction.max}")
}

enum class Operation {
  INC {
    override fun apply(x: Int, y: Int): Int = x + y
  },

  DEC {
    override fun apply(x: Int, y: Int): Int = x - y
  },

  GT {
    override fun apply(x: Int, y: Int): Boolean = x > y
  },

  GTEQ {
    override fun apply(x: Int, y: Int): Boolean = x >= y
  },

  LT {
    override fun apply(x: Int, y: Int): Boolean = x < y
  },

  LTEQ {
    override fun apply(x: Int, y: Int): Boolean = x <= y
  },

  EQ {
    override fun apply(x: Int, y: Int): Boolean = x == y
  },

  NEQ {
    override fun apply(x: Int, y: Int): Boolean = x != y
  };

  abstract fun apply(x: Int, y: Int): Any
}

val supportedOperations = mapOf(
    "inc" to INC,
    "dec" to DEC,
    ">" to GT,
    ">=" to GTEQ,
    "<" to LT,
    "<=" to LTEQ,
    "==" to EQ,
    "!=" to NEQ
)

class Instruction(
    private val name: String,
    private val operation: Operation,
    private val amount: Int,
    private val condition: () -> Boolean
) {

  fun execute(value: Int = register[name]!!) {
    if (condition()) {
      val new = operation.apply(value, amount) as Int
      register.put(name, new)
      max = max(max, new)
    }
  }

  companion object {
    val register: MutableMap<String, Int> = mutableMapOf()

    var max: Int = Int.MIN_VALUE

    fun parse(input: String): List<Instruction> {
      register.clear()

      return input
          .split('\n')
          .filter(String::isNotEmpty)
          .map {
            val elements = it.split(' ')

            val name = elements[0]
            val operation = supportedOperations[elements[1]]!!
            val amount = elements[2].toInt()
            val registerValue: (String) -> Int = { register[it]!! }
            val conditionOperation = supportedOperations[elements[5]]!!
            val conditionValue = elements[6].toInt()
            val condition: () -> Boolean = {
              conditionOperation.apply(registerValue(elements[4]), conditionValue) as Boolean
            }

            register.putIfAbsent(name, 0)
            Instruction(name, operation, amount, condition)
          }
    }
  }
}
//</editor-fold>

//<editor-fold desc="Day 9">
val testInputsDay9Part1 = listOf(
    "{}",
    "{{{}}}",
    "{{},{}}",
    "{{{},{},{{}}}}",
    "{<a>,<a>,<a>,<a>}",
    "{{<ab>},{<ab>},{<ab>},{<ab>}}",
    "{{<!!>},{<!!>},{<!!>},{<!!>}}",
    "{{<a!>},{<a!>},{<a!>},{<ab>}}"
)

val testInputsDay9Part2 = listOf(
    "<>",
    "<random characters>",
    "<<<<>",
    "<{!>}>",
    "<!!>",
    "<!!!>>",
    "<{o\"i!a,<{i<a>"
)

val inputDay9 = FileProvider.url("/inputDay9.txt")!!.readText().trim()

fun testDay9Part1() = testInputsDay9Part1.forEach {
  println("$it = ${scoreGarbageGroups(it).first}")
}

fun testDay9Part2() = testInputsDay9Part2.forEach {
  println("$it = ${scoreGarbageGroups(it).second}")
}

fun day9Part1And2() = println(scoreGarbageGroups(inputDay9))

fun scoreGarbageGroups(input: String): Pair<Int, Int> {
  var level = 0
  var score = 0
  var numGarbage = 0
  var inGarbage = false

  input.forEachIndexed { index, char ->
    if (inGarbage) {
      if (input.take(index).takeLastWhile { it == '!' }.count() % 2 == 0) {
        when (char) {
          '>' -> inGarbage = false
          '!' -> { // NOOP
          }
          else -> numGarbage++
        }
      }
    } else {
      when (char) {
        '{' -> level++
        '}' -> score += level--
        '<' -> inGarbage = true
        else -> { // NOOP
        }
      }
    }
  }

  return score to numGarbage
}
//</editor-fold>

//<editor-fold desc="Day 10">
fun <T> testAssert(expected: T, actual: T) =
    assert(expected == actual) { "Expected: $expected, but was: $actual" }

val testSeedDay10 = (0..4).toList()

val seedDay10 = (0..255).toList()

val testInputDay10 = listOf(3, 4, 1, 5)

val inputDay10 = "70,66,255,2,48,0,54,48,80,141,244,254,160,108,1,41"

val inputDay10Part1 = inputDay10.split(',').map(String::toInt)

val day10Part1Function = { hash: List<Int> -> hash[0] * hash[1] }

fun testDay10Part1() = println(knotHash(
    lengths = testInputDay10,
    seed = testSeedDay10,
    function = day10Part1Function
))

fun day10Part1() = println(knotHash(
    lengths = inputDay10Part1,
    seed = seedDay10,
    function = day10Part1Function
))

val day10Part2Suffix = listOf(17, 31, 73, 47, 23)

fun prepDay10Part2Input(input: String) = input.map(Char::toInt) + day10Part2Suffix

val testInputDay10Part2 = prepDay10Part2Input("1,2,3")

fun testDay10Part2Input() = testAssert(
    listOf(49, 44, 50, 44, 51, 17, 31, 73, 47, 23),
    testInputDay10Part2
)

val inputDay10Part2 = prepDay10Part2Input(inputDay10)

val sparseToDenseHash = { window: List<Int> ->
  window.drop(1).fold(window.first()) { acc, i -> acc xor i }
}

fun testSparseToDenseHash() {
  testAssert(64, sparseToDenseHash(listOf(65, 27, 9, 1, 4, 3, 40, 50, 91, 7, 6, 0, 2, 5, 68, 22)))
}

val day10Part2Function = { hash: List<Int> ->
  hash
      .chunked(16, sparseToDenseHash)
      .flatMap { it.toString(16).run { if (length == 1) "0" + this else this }.asIterable() }
      .joinToString("")
}

val testDay10Part2HashValidatorInputs = listOf("", "AoC 2017", "1,2,3", "1,2,4")

val testDay10Part2HashValidatorOutputs = listOf(
    "a2582a3a0e66e6e86e3812dcb672a272",
    "33efeb34ea91902bb2f59c9920caa6cd",
    "3efbe78a8d82f29979031a4aa0b16a9d",
    "63960835bcdc130f0b66d7ff4f6a5a8e"
)

val day10Part2Rounds = 64

fun testDay10Part2() =
    (testDay10Part2HashValidatorOutputs zip testDay10Part2HashValidatorInputs)
        .forEach {
          testAssert(
              it.first,
              knotHash(
                  prepDay10Part2Input(it.second),
                  seedDay10,
                  day10Part2Rounds,
                  day10Part2Function
              )
          )
        }

fun testDay10Part2Tests() {
  testDay10Part2Input()
  testSparseToDenseHash()
  testDay10Part2()
}

fun day10Part2() =
    println(knotHash(inputDay10Part2, seedDay10, day10Part2Rounds, day10Part2Function))

fun <T> Iterable<T>.cycle(): Sequence<T> = generateSequence { this }.flatten()

fun <T, V> knotHash(
    lengths: List<Int>,
    seed: List<T>,
    rounds: Int = 1,
    function: (List<T>) -> V
): V {
  var hash = seed
  val n = seed.size
  var index = 0
  var skip = 0

  for (ignored in 1..rounds) {
    lengths.forEach { length ->
      hash = hash
          .cycle()
          .drop(index)
          .take(length)
          .asIterable()
          .reversed()
          .plus(hash.cycle().drop(index).take(n).drop(length))
          .cycle()
          .drop(n - index % n)
          .take(n)
          .toList()
      index += length + skip++
    }
  }

  return function(hash)
}
//</editor-fold>

//<editor-fold desc="Day 11">
val testInputDay11 = listOf(
    "ne,ne,ne",
    "ne,ne,sw,sw",
    "ne,ne,s,s",
    "se,sw,se,sw,sw"
)

val testOutputDay11 = listOf(3, 0, 2, 3)

fun testDay11Part1() =
    testInputDay11.forEachIndexed { index, path ->
      testAssert(testOutputDay11[index], hexLength(path))
    }

val inputDay11 = FileProvider.url("/inputDay11.txt")!!.readText()

fun day11Part1() = println(hexLength(inputDay11))

fun hexLength(path: String): Int =
    path
        .trim()
        .toUpperCase()
        .split(',')
        .map { HexDirection.valueOf(it) }
        .fold(mutableListOf()) { shortestPath: MutableList<HexDirection>, direction: HexDirection ->
          shortestPath.apply {
            val opposite = direction.opposite()
            if (!remove(opposite)) {
              val canceller = opposite.neighbors().singleOrNull { it in shortestPath }
              if (canceller == null) {
                add(direction)
              } else {
                remove(canceller)
                add(HexDirection.values().single {
                  it.neighbors().containsAll(setOf(direction, canceller))
                })
              }
            }
          }
        }
        .size

enum class HexDirection {
  N {
    override fun opposite() = S
    override fun neighbors() = setOf(NW, NE)
  },
  NE {
    override fun opposite() = SW
    override fun neighbors() = setOf(N, SE)
  },
  SE {
    override fun opposite() = NW
    override fun neighbors() = setOf(NE, S)
  },
  S {
    override fun opposite() = N
    override fun neighbors() = setOf(SE, SW)
  },
  SW {
    override fun opposite() = NE
    override fun neighbors() = setOf(S, NW)
  },
  NW {
    override fun opposite() = SE
    override fun neighbors() = setOf(SW, N)

  };

  abstract fun opposite(): HexDirection
  abstract fun neighbors(): Set<HexDirection>
}
//</editor-fold>
