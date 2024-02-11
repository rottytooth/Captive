# Captive

A language where every text is a valid program

Inspired by the Oulipian constraint set *Prisoner's Constraint*. Every lower-case letter that reaches above the x-height or below the baseline is a command, all other letters are ignored.

## Commands

|1st |2nd |cmd|followed by|
|---|---|---|---|
|l||push|int to push to the stack|
|p|k|pop|index of item to pop|
|p|h|not||
|p|t|if||
|p|d|end||
|t|t|emit||
|t|h|rot|int for how many to rotate|
|f||sub||
|b||mul||
|d|d|add||
|d|g|>||
|k||div||
|q||mod||
|g||while||
|j||==||
|h||dup||
|y||end||

## Constants

Ints are marked by a string of letters indicating which value to add, starting from zero. They are always emitted as Unicode. To get floating point numbers, one must load to numbers to the stack and divide.

|ltr|value|
|---|---|
|q|subtract next value|
|k|subtract next value|
|f|subtract next value|
|b|subtract next value|
|d|64|
|g|32|
|j|24|
|h|16|
|p|8|
|l|4|
|t|1|
|y|end of const|

## Captive Interpreter 0.1
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