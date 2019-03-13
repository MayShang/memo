## ways to think
1. in imperative (cmd) language you get things done by giving computer a sequence of tasks and then it executes them. When executing them, it can change state. you have control flow structure for doing some action several times. while in purely functional programming you don't tell computer what to do as such but rather you tell it what stuff is.

2. the set contains the doubles of all natural numbers that satisfy the predicate.

3. `:t` to check type. HS types are written in capital case.
`::` means `is type of` so we use to make type declarations for functions `funcName :: Int -> Int`

4. types: `Bool` `Int` `Float` `Double` `Char` `String`

5. `polymorphic functions` `:t head` => [a]->a

6. `Typeclasses` is a sort of interface that defines some behavior.  
if a type is a part of a typeclass, that means it supports and implements the behavior the typeclass describes.

7. concept: 
`:t (==)` output: (==) :: (Eq a) => a -> a -> Bool
the equality function takes any two values that are of the same type and returns a Bool. the type of those two values must be a member of the Eq class
(Eq a) is class contrain. so a is member of this class, and has or implement Eq behavior.
`Eq` is used for types that support equality testing. the functions its members implement are `==` and `/=`. so if there is an Eq class constraint for a type variable in a function, it uses `==` and `/=` somewhere inside its definition. above types are part of Eq, so they can be tested for equality.

## platform
* ghci
* ":l myfunctions" => call myfunctions.hs
* "quit"
