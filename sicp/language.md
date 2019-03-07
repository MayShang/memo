# language mechanisms
* expressions
* means of combination
* means of abstraction

why these three elements matter?

* expressions, how to touch it, how to play with it.
* combination ways, after play with its basic, how to apply to implement player's idea.
* abstraction ways, generalization, used for far field, broadcast

another idea:
two elements: procedures and data. actually they are really no distinct.

how to understand this idea? 
procedure is data, why I never know this entities.

data is 'stuff' that we want to manipulate, and procedures aer descriptions of the rules for manipulating the data. Thus, any powerful programming language should be able to describe primitive data and primitive procedures and should have methods for combining and adstracting procedures and data.


## building block
compound procedures consist of building block
Interpreter evaluates the elements of the combination and applies the procedure (which is the value of the operation) to the arguments ( which are the values of the operands of the combination ).
we can assume that the mechanism for applying primitive procedures to arguments is 
to Apply a compond procedure to arguments, evaluate the body of the procedure with each formal parameter replaced by the correspoding arguments.

* declarative or imperative

* really important idea
1. how computer computer square roots(p62)(p68)
break into a number of subproblems: 
  * how to tell whether a guess is good enough
  * how to improve a guess
  * and so

2. (p68) we must learn to visualize the processes generated by various types of procedures. only after we have developed such a skill can we learn to reliably construct programs that exhibit the desired behavior.
'A procedure is a pattern for the local evolution of a computational process.'

3. (p72) iterative process
an iterative process includes a fixed number of state variables, together with a fixed rule that describe how the state variable should be updated as the process moves from state to state and an optional, end test that specifies conditions under which the process should terminate.

4. (p73) recursive process vs. recursive procedure
recursive procedure is a syntax of how a procedure is written. the process can be recursive and iterative process.
