package io.github.erikhuizinga.adventofcode

import java.net.URL
import kotlin.math.*

/** Created by erik.huizinga on 27-12-17 */
fun main(args: Array<String>) {
  day4Part1()
}

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

val testInputsDay4 = listOf("aa bb cc dd ee", "aa bb cc dd aa", "aa bb cc dd aaa")

val inputDay4File = FileProvider.url("/inputDay4.txt")

object FileProvider {
  fun url(name: String): URL? = javaClass.getResource(name)
}

fun testDay4Part1() = testInputsDay4.forEach { println(it.countValidPassphrases()) }

fun day4Part1() = println(inputDay4File?.readText()?.countValidPassphrases())

private fun String.countValidPassphrases(): Int =
    split("\n").filter { it.isNotEmpty() }.count { it.isValidPassphrase() }

private fun String.isValidPassphrase(): Boolean = split(" ").run { size == toSet().size }