from argparse import ArgumentParser
import os
import re
import numpy as np

class lexer:

    COMMANDS = {
        "l": "push",
        "p": {
            "k": "pop",
            "h": "not",
            "t": "if",
            "d": "end"
        },
        "t": {
            "t": "emit",
            "h": "rot"
        },
        "f": "sub",
        "b": "mul",
        "d": {
            "d": "add",
            "g": "gt"
        },
        "k": "div",
        "q": "mod",
        "g": "while",
        "j": "eq",
        "h": "dup",
        "y": "end"
    }

    NUMBERS = {
        "q": "-",
        "k": "-",
        "f": "-",
        "b": "-",
        "d": 64,
        "g": 32,
        "j": 24,
        "h": 16,
        "p": 8,
        "l": 4,
        "t": 1,
        "y": "end"
    }

    def lex_number(self, input_txt, idx):
        subtract = False
        curr_number = 0

        while (idx < len(input_txt)):
            numcode = lexer.NUMBERS[input_txt[idx]]
            if numcode == "-":
                subtract = True
            elif numcode == "end":
                return curr_number, idx
            elif subtract:
                if numcode != "-":
                    # subsequent subtract markers are ignored
                    curr_number -= numcode
                subtract = False
            else:
                curr_number += numcode
            idx += 1

        return curr_number, idx

    def strip_non_commands(self, input_txt):
        return re.sub("[^bdghjklpqty]", "", input_txt)
    
    def lex(self, input_txt):
        command_text = self.strip_non_commands(input_txt)

        program = []
        idx = 0
        first = None

        while idx < len(command_text):
            c = command_text[idx]

            if first:
                if c in first:
                    cmd = first[c]
                    program.append(cmd)
                    idx += 1
                    first = None

                    if cmd == "rot" or cmd == "pop":
                        number, idx = self.lex_number(command_text, idx+1)
                        program.append(number)
                    continue
                # if the second letter is incompatible with the first, the first is ignored and the second is treated as its own command
                first = None

            cmd = lexer.COMMANDS[c]

            if not isinstance(cmd, str):
                # load the possibilities, to determine in next iteration
                first = cmd
            else:
                program.append(cmd)

            if cmd == "push":
                number, idx = self.lex_number(command_text, idx+1)
                program.append(number)
            idx += 1
        
        return program


class interpreter:
    def __init__(self):
        self.lexer = lexer()

    def parse(self, nodes, verbose=False):
        # FIXME: passing verbose all over the place is ridiculous. This should use standard logging
        idx = 0
        stack = []
        cmd_stack = []

        retset = self.parse_nodes(nodes, idx, stack, cmd_stack, verbose=verbose)
        return retset[3], stack

    def parse_nodes(self, nodes, idx, stack, cmd_stack, output = "", verbose=False):

        # where we came in (in case this is a while loop)
        starting_idx = idx

        # tracking for the simplest infinite loop
        stack_cache = []

        while idx < len(nodes):

            n = nodes[idx]
            
            if len(cmd_stack) > 0 and cmd_stack[-1] == "skip":
                if n == "end":
                    cmd_stack.pop()
                else:
                    pass
                    # what happens here is we skip all the elifs below until the end is reached
            elif n == "end":
                # at the end of an if, we'll always pop
                if len(cmd_stack) > 0 and cmd_stack[-1] == "if":
                    cmd_stack.pop()
                # for while, we have to test the top item
                if len(cmd_stack) > 0 and cmd_stack[-1] == "while":

                    # infinite loop handling
                    if len(stack_cache) == 0:
                        stack_cache.append(stack)
                    elif stack in stack_cache:
                        # if the stack is in exactly the same state as a previous iteration, we consider it an infinite loop and exit
                        cmd_stack.pop()
                        return idx, stack, cmd_stack, output
                    else:
                        if verbose:
                            # show the comparison between the two
                            for old_stack in stack_cache:
                                if len(old_stack) != len(stack):
                                    print(f"cached stack len: {len(old_stack)}, curr stack length: {len(stack)}")
                                else:
                                    #FIXME: rework to avoid this conversion
                                    print(np.subtract(np.asarray(old_stack), np.asarray(stack)))
                        stack_cache.append(stack)

                    # actual testing of the while condition
                    # if there's nothing in the stack, we also return
                    if len(stack) == 0 or stack[-1] <= 0:
                        cmd_stack.pop()
                        return idx, stack, cmd_stack, output
                    else:
                        idx = starting_idx
            elif n == "push":
                if idx + 1 < len(nodes) and isinstance(nodes[idx+1], int):
                    stack.append(nodes[idx+1])
                    idx += 1
            elif n == "rot":
                if idx + 1 < len(nodes) and isinstance(nodes[idx+1], int):
                    if len(stack) > 0:
                        modded_size = len(stack) - (nodes[idx+1] % len(stack))
                        stack = stack[modded_size:] + stack[:modded_size]
                    idx += 1
            elif n == "pop":
                if idx + 1 < len(nodes) and isinstance(nodes[idx+1], int) and len(stack) > 0:
                    topop = nodes[idx+1] % len(stack)
                    stack.pop(topop)
                    idx += 1
            elif n == "dup":
                if len(stack) > 0:
                    stack.append(stack[-1])
            elif n == "not":
                if stack[-1] == 0:
                    stack[-1] == 1
                else:
                    stack[-1] = 0 - stack[-1]
            elif n == "if":
                if stack[-1] > 0:
                    cmd_stack.append("if")
                else:
                    cmd_stack.append("skip")
            elif n == "emit":
                if len(stack) > 0:
                    output += chr(round(stack[-1])) # to print, we round to closest int
                    stack.pop()
            elif n == "sub":
                if len(stack) > 1:
                    stack.append(stack.pop() - stack.pop())
            elif n == "mul":
                if len(stack) > 1:
                    stack.append(stack.pop() * stack.pop())
            elif n == "add":
                if len(stack) > 1:
                    stack.append(stack.pop() + stack.pop())
            elif n == "div":
                if len(stack) > 1:
                    num = stack.pop()
                    den = stack.pop()
                    if den == 0:
                        stack.append(0)
                    else:
                        stack.append(num / den)
            elif n == "gt":
                if len(stack) > 1:
                    stack.append(int(stack.pop() > stack.pop()))
            elif n == "eq":
                if len(stack) > 1:
                    stack.append(int(stack.pop() == stack.pop()))
            elif n == "while":
                if stack[-1] > 0:
                    cmd_stack.append("while")
                    idx, stack, cmd_stack, output = self.parse_nodes(nodes, idx + 1, stack, cmd_stack, output=output, verbose=verbose)
                else:
                    cmd_stack.append("skip")

            idx += 1

        return idx, stack, cmd_stack, output

    def interpret_file(self, file_loc, codefile=None, outfile=None, includestack=False, verbose=False):

        if not os.path.exists("out"):       
            os.makedirs("out") 

        with open(file_loc, "r", encoding="utf-8") as file:
            content = file.read()

        lexnodes = self.lexer.lex(content)
        if verbose:
            print(lexnodes)

        if codefile:
            if (os.path.isfile(codefile)):
                os.remove(codefile)

            with open(codefile, "x", encoding="utf-8") as o:
                o.write('\n'.join(str(x) for x in lexnodes))

        output, stack = self.parse(lexnodes, verbose=verbose)

        if outfile:
            if (os.path.isfile(outfile)):
                os.remove(outfile)

            with open(outfile, "x", encoding="utf-8") as o:
                o.write(output)

                if includestack:
                    o.write("\n\nSTACK CONTENTS:")
                    o.write(str(stack))

        return output



if __name__ == "__main__":

    parser = ArgumentParser(description='Captive Interpreter 0.1', 
                            epilog='More at https://danieltemkin.com/Esolangs/Captive')

    parser.add_argument('progfile', metavar='progfile', type=str, 
                        help='Captive program file')
    parser.add_argument('--out', dest='outfile', default=None, 
                        help='where to write output from the program')
    parser.add_argument('--ps', dest='inclex', default=False, 
                        help='where to write the program transpiled to stack pseudocode')
    parser.add_argument('-s', dest='incstack', action='store_true', 
                        default=False, help='include final state of stack in output')
    parser.add_argument('-v', dest='verbose', action='store_true', 
                        default=False, help='verbose logging')
    args = parser.parse_args()

    intr = interpreter()
    result = intr.interpret_file(args.progfile, args.inclex, args.outfile, 
                                 args.incstack, args.verbose)

    if not args.outfile or args.verbose:
        print(result)
