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
1. what does it mean?
`(->) r a` can be written as `r -> a`

2. defination
```
instance Functor ((->) r) where
    fmap f g = (\x -> f (g x))
```

```
instance Functor ((->) r) where
    fmap = (.)
```
3. how to use it
```
fmap (*3) (+100) 1  -- 303
(*3) `fmap` (+100) $ 1
(*3) . (+100) $ 1
```

4. functor laws
if we can show that type obeys both functor laws, we can rely on it having the same functional behaviors as other functors when it comes to mapping. we can know that when we use fmap on it, there won't be anything other than mapping going on behind the scenes and that it will act like a thing that can be mapped over.

```
instance Functor CMaybe where
    fmap f CNothing = CNothing
    fmap f (CJust x y) = CJust x (f y)
```
```
fmap (++"hwlo") (CJust 3 "ei")
```
5. apply function over functor
functor is container, has values inside it.
function is normal function, apply a normal function over functor, meaning, how to apply normal function over wrapped values.
so functor is a noun and works on data type.

## Applicative Functor
1. `Control.Applicative`
2. To work on values inside it and a function inside it too, how to apply this wrapped function on the wrapped values.

3. defination
```
class (Functor f) => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b
```
what is `f`? applicative functor instance. really interesting, not a function, not like functor defination, `f` means normal function. so this means that takes a applicative(functor) type function, and value inside the functor, return value still inside this functor.  f of  `f (a -> b)` could be different from `f` of `f a` and `f b` but f inside `f a` and `f b` are the same.
so pure means take a value of any type and return an applicative functor with that value inside it.

`<*>` takes a functor that has a function in it, and another functor and sort of extracts that function from the first functor and then maps it over the second one.

4. `Maybe` instances
```
instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    (Just f) <*> something = fmap f something
```

5. cases
```
Just (+3) <*> Just 9
```

6. Applicative Specific
allow to operate on several functors with a single function.
```
pure (+) <*> Just 3 <*> Just 5
```

important: `pure f <*> x` equal to `fmap f x`
so `pure f <*> x <*> y <*> ...` means `fmap f x <*> y <*> ...`
so from this, we know why defination `<$>` is
```
(<$>) :: (Functor f) => (a -> b) -> f a -> f b
f <$> x = fmap f x
```

then `pure f <*> x <*> y <*> ...` could be written as `f <$> x <*> y <*> ...`

```
pure (+) <*> Just 3 <*> Just 5
```

can be written as

```
(+) <$> Just 3 < *> Just 5
```

7. List instance
```
instance Applicative [] where
    pure x = []
    fs <*> xs = [f x | f <- fs, x <- xs]
```
this list impl is pretty cool! `<-` means `unwrap`.

example:
```
(++) <$> ["he", "wro"] <*> ["wr", "ere", "ereef"]
[(*0), (+23), (^2)] <*> [1, 2, 3]
[(*), (+)] <*> [2, 3] <*> [4, 5]
```

```
(*) <*> [2, 3] <*> [3, 4]

```
impl the product operation
```
[2
 3] [3 4]
```

```
(++) <$> ["ha", "heh"] <*> ["?", "!"]  -- ["ha?", "ha!", "heh?", "heh!"]
```

```
filter (>50) $ (*) <$> [2, 5, 10] <*> [5, 10, 11]
```

8. IO instance
```
instance Applicative IO where
    pure = return
    a <*> b = do 
        f <- a
        x <- b
        return (f x)
```

```
(<*>) :: IO (a -> b) -> IO a -> IO b
```
it takes an IO action that yield a function as its result and another IO action and create a new IO action from those two that.

```
myAction :: IO String
myAction = do
    a <- getLine
    b <- getLine
    return $ a ++ b
```
impl by applicative style:
```
myAction = (++) <$> getLine <*> getLine

```
make more concise and terse code:
```
main = do
    a <- (++) <$> getLine <*> getLine
    putStrLn $ "retur is " ++ a

```

when you find yourself binding some IO actions to names and then calling some functions on them and presenting that as the result by using return, consider using applicative style.
this means if 
 - bind names
 - apply func on this name
 - return result
use applicative style.

9. `(->) r` instance
```
instance Applicative ((->) r) where
    pure x = (\_ -> x)
    f <*> g = \x -> f x (g x)
```
pure means a minimal default context still yield that value as a result.
```
(pure 3) "bls" -- 3

(+) <$> (+3) <*> (*100) $ 5
```
this means `5` first applied to `(+3)` and `(*100)`, resulting in `8` and `500`, then (+) apply to 8 and 500

```
(\x y z -> [x, y, z]) <$> (+3) <*> (*2) <*> (/2) $ 4 -- [x, y, z]
```

10. `ZipList` instance
```
instance Applicative ZipList where
    pure x = ZipList (repeat x)
    ZipList fs <*> ZipList xs = ZipList (zipWith (\f x -> f x) fs xs)
```
it apply the first function to the first value, and the sec func to the sec value, etc, because `zipWith` working.
use `getZipList` to extract a raw list out of a zip list.
```
getZipList $ (+) <$> ZipList [1, 2, 3] <*> ZipList [100, 200, 300] -> [101, 202, 303]
getZipList $ (,,) <$> ZipList "dog" <*> ZipList "fish" <*> ZipList "rat" --[('d','f','r'), ('o', 'i', "a", ..)]
```
`(, ,)` equal to (\x y z -> (x, y, z)),  `(,)` equal to `\x y -> (x, y)`

11. "liftA2"
```
liftA2 :: (Applicative f) => (a -> b -> c) -> f a -> f b -> f c
liftA2 f a b = f <$> a <*> b
```
we can take two applicative functors, and combine them into one applicative functor that has inside it the result of those two applicative functors in a list.
for example, we have `Just 3` and `Just 4` two functor, how to combine them?

```
(:) (Just 3) (Just [4])  -- couldn't work, why? (`:` is normal func, you can't use it apply to functor Maybe)

```
so you have to LIFT this function using `liftA2`

and 
```
(:) <$> (Just 3 ) <*> (Just [3]) --working
```
and 
```
liftA2 (:) (Just 3 ) (Just [3]) --working
```

12. `sequenceA`
takes a list of applicatives and returns an applicative that has a list as its result value.
```
sequenceA :: (Applicative f) => [f a] -> f [a]
sequenceA [] = pure []
sequenceA (x:xs) = (:) <$> x <*> sequenceA xs
```

sequenceA mainly works on list.
`sequenceA [(+2), (*10)]` means `(:) <$> (+2) <*> (*10)`

sequenceA is cool when we have a list of functions and we want to feed the same input to all of them and then view the list of results.
```
map (\f -> f 7) [(>4), (<10), odd]
sequenceA [(>4), (<10), odd] 7
and $ sequenceA [(>4), (<10), odd] 7  -- AND operation
```

how to understand `sequenceA [[1, 2], [3, 4]]]`
 - eval to `(:) <$> [1, 2] <*> sequenceA [[3, 4]]`
 - eval the inner sequenceA further, `(:) <$> [1, 2] <*> ((:) <$> [3, 4] <*> sequenceA [])`
 - then we reached the edge condition, `(:) <$> [1, 2] <*> ((:) <$> [3, 4] <*> [[]])`
 - now eval `((:) <$> [3, 4] <*> [[]])` part, resulting in `[3:[], 4:[]]` equal to `[[3], [4]]`, so now we have `(:) <$> [[3], [4]]`
 - eval, now `:` is used with every possible value from the left list (1 and 2) with every possible value in the right list [3] and [4], which results in [1:[3], 1:[4], 2:[3], 2:[4]], which is [[1,3], [1,4], [2, 3], [2,4]]

for IO actions, sequenceA is the same thing as sequence.
it takes a list of IO actions and returns an IO action.

```
sequenceA [getLine, getLine, getLine]  -- ["her", "weri", "wer"]
```

## data type
### difference between `data` and `newtype`
when using `newtype`, you're restricted to just one constructor with one field.

