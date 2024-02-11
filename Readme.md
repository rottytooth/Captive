# Captive

A language where every text is a valid program

Inspired by the Oulipian constraint set *Prisoner's Constraint*

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