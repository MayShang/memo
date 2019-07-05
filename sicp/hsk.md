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
```
data I a = I a
instance Functor I where
    fmap f I x = I (f x)

-- how to use it?
fmap (+4) (I 5)  -- I 9
fmap (++ "hello ") (I "world ")
fmap ("hello " ++) (I "world ") -- I "hello world"

```

really funny
but still not sure how to define a useful data type

initution about funcntor
give me a function that takes an `a` and return a `b`, and a box with `a`
inside it and I'll give you box with a `b` side it. it kind of applies
the function to the element inside the box.
what does it mean?
you give me a function and a box, I'll give a box, this box includes
the result of this function.

so functor keeps shape still, but manipulates its contents.

how to make an instance of functor?
one:
  to make what to be a functor?
  type contructure !!
  so it's data type we are going to work on.
two:
  this type contructure has to have a kind of `*->*`
  what does it mean?
  this type contructure has to take exactly one concrete type as a
  type parameter.
  for example, `Maybe` takes one type parameter to produce a concrete one.
  so `Maybe Int` `Maybe String` working
  what about two parameter?
  partially apply the type constructure until it only take one type parameter.
  so `instance Functor Either where` wrong, you have to write
  `instance Functor (Either a) where`, but how to work on the first parameter?

extension:
what about IO Functor
```
instance Functor IO where
    fmap f action = do
        result <- action
        return (f result)
```
mapping sth over an IO action will be an IO action. 

```
main = do line <- fmap reverse getLine
    print $ line
```

```
main = do line <- fmap (intersperse '-' . reverse . map toUpper) getLine
    print $ line
```

### come to know `(->) r` functor

