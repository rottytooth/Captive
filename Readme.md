# Captive

A language where every text is a valid program

Inspired by the Oulipian constraint set *Prisoner's Constraint*. Every lower-case letter that reaches above the x-height or below the baseline is a command, all other letters are ignored.

More at https://danieltemkin.com/Esolangs/Captive

## Reading a text as a program

The command letters are `b d f g h j k l p q t y`. Everything else is ignored: other letters, punctuation, spaces, and capitals. A capital is never a command, so *The* reads only as `h`.

Commands are read left to right. `p`, `d` and `t` begin two-letter commands, and a pair is taken whenever the next letter completes one. A `t` that doesn't begin a pair is emit; a `p` or `d` that doesn't is dropped, and the next letter is read on its own.

For example, [hi.txt](programs/hi.txt):

> He could expect. They overlap, and every artist is at it.

reads as `ldpthy lpdy t t t t`: push 105, push 72, emit, emit, emit, emit. It prints `Hi`; the last two emits find the stack empty and do nothing. The capital *T* in *They* isn't a command, so its *h* and *y* finish the first number rather than making a swap.

## Commands

Captive is stack-based.

|1st |2nd |cmd|followed by|
|---|---|---|---|
|l||push|number to push to the stack|
|t|h|swap||
|t||emit||
|h||dup||
|d|t|over||
|p|p|rot 1: the top item to the bottom||
|d|p|rot −1: the bottom item to the top||
|p|g|rot|number of places to rotate|
|p|l|pop|index of item to pop|
|p|h|not||
|p|t|if||
|p|d|end||
|d|d|add||
|d|g|>||
|f||sub||
|b||mul||
|k||div||
|q||mod||
|g||while||
|j||==||
|y||end||

The commands programs use most sit on the letters and pairs English uses most: `th`, the commonest pair in English, is swap, and rotating by one, by far the commonest rotation, needs no number. This keeps texts that spell real programs from leaning on the few words that fit; see [Changes in 0.2](#changes-in-02).

- **Arithmetic and comparisons** (`f` `b` `dd` `k` `q` `dg` `j`) pop the top two items and push the result. The top item is the left-hand side: sub pushes *top − second*, div *top ÷ second*, and > pushes 1 if *top > second*, otherwise 0. `==` pushes 1 if the two are equal, otherwise 0.
- **div** and **mod** same, but push 0 when dividing by zero.
- **emit** pops the top item and prints it as a character: rounded to a whole number, made positive, and wrapped into the ASCII range (values above 127 are taken mod 127).
- **pop** removes the item at the given index, counting from the bottom of the stack (0 is the bottom) and wrapping around past the top.
- **swap** exchanges the top two items, and **over** pushes a copy of the second item.
- **rot** rotates the stack: the top *n* items move to the bottom, or with a negative *n*, the bottom items to the top.
- **emit then dup**: `th` reads as swap, so to emit and then dup, put a `d` between them (`tdh`); it begins no command, so it's dropped.
- **not** turns 0 into 1 and negates anything else, so a true (positive) value becomes false.
- **if** runs its block, up to the matching end, if the top item is positive, and skips it otherwise. **while** repeats its block as long as the top item is positive. Neither pops the value it tests, and an empty stack counts as false. Blocks nest, and either `y` or `pd` ends one.
- **Infinite loops:** a while loop also stops when the stack returns to a state from an earlier pass, since the loop would otherwise never end. A loop that grows the stack forever isn't caught yet, but may be in future versions.
- A command that needs more items than the stack holds does nothing.

## Constants

Numbers follow push (`l`), pop (`pl`) and rot (`pg`). A number is a string of letters ending in `y`. Starting from zero, each letter adds its value, and `q`, `k`, `f` and `b` subtract the next value instead: `l dhgt y` pushes 64 + 32 + 4 + 1 = 101, and `l dft y` pushes 64 − 1 = 63. Inside a number, every command letter is part of the number; commands resume after the `y`.

The values are binary places, so every number from 0 to 127 can be spelled with each letter at most once, and letters can repeat for other spellings: `l lll y` and `l dfl y` both push 48.

To get floating point numbers, load two numbers to the stack and divide.

|ltr|value|
|---|---|
|q|subtract next value|
|k|subtract next value|
|f|subtract next value|
|b|subtract next value|
|d|64|
|h|32|
|l|16|
|p|8|
|g|4|
|j|2|
|t|1|
|y|end of const|

## Changes in 0.2

Version 0.2 is a re-working of several commands to make it more natural to English words. Basically, I got sick of my Captive program generator using the word "property" or "proprietary" over and over.

|cmd|0.1|0.2|
|---|---|---|
|swap|none|`th`|
|over|none|`dt`|
|rot|`pp` or `dp`, then a number|`pp` (rot 1), `dp` (rot −1), or `pg` then a number|
|pop|`pk`, then a number|`pl`, then a number|

The number letters were revalued as well: in 0.1 they were d 64, g 32, j 24, h 16, p 8, l 4, t 1, and in 0.2 they are d 64, h 32, l 16, p 8, g 4, j 2, t 1.


## Captive Interpreter 0.2

```
python cap_interpreter.py programs/hello_world.txt
```

```
positional arguments:
    progfile       Captive program file

options:
  -h, --help     show this help message and exit
  --out OUTFILE  where to write output from the program
  --ps INCLEX    where to write the program transpiled to stack pseudocode
  -s             include final state of stack in output
  -v             verbose logging
```

Sample programs are in [programs/](programs/).

Run the tests with `python -m pytest`.
