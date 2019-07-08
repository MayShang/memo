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


## one practical example
```
type Birds = Int
type Pole = (Birds, Birds)

landLeft :: Birds -> Pole -> Pole
landLeft n (left, right) = (left + n, right)

landRight :: Birds -> Pole -> Pole
landRight n (left, right) = (left, right + n)

landLeft 3 (landRight 4 (landLeft 4 (0, 0)))
```
this is the lexiel transfermation.

`Maybe` Impl
```
landLeft :: Birds -> Pole -> Maybe Pole
landLeft n (left, right)
    | abs ((left +n) - right) < 4 = Just (left + n, right)
    | otherwise                   = Nothing

landRight :: Birds -> Pole -> Maybe Pole
landRight n (left, right)
    | abs (left - (right + n)) < 4 = Just (left, right + n)
    | otherwise                    = Nothing

```
because `Maybe` is monad, so we can use landLeft/right as monad instance directly!!
```
landLeft 1 (0, 0) >>= landRight 3
return (0, 0) >>= landRight 3 >>= landLeft 2 >>= landRight 4
```

## do-block
1. examples

`case` chain
```
routine :: Maybe Pole
routine = case landLeft 1 (0, 0) of
    Nothing -> Nothing
    Just Pole1 -> case landRight 4 Pole1 of
        Nothing -> Nothing
        Just Pole2 -> case landLeft 2 Pole2 of
            Nothing -> Nothing
            Just Pole3 -> landLeft 1 Pole3

```
do-block impl
```
routine :: Maybe Pole
routine = do
    start <- return (0, 0)
    first <- landLeft 2 start
    sec   <- landRight 3 first
    landLeft 1 sec
```

action chain primitive impl
```
foo :: Maybe String
foo = Just 3 >>= (\x ->
      Just "!" >>= (\y ->
      Just (show x ++ y)))
```

do block impl
```
foo' :: Maybe String
foo' = do
    x <- Just 3
    y <- Just "!"
    Just (show x ++ y)
```

2. understanding do-block
in do notation, when we bind monadic values to name, we can utilize pattern matching, just like `let` expressions and function parameters.
```
justH :: Maybe [Char]
justH = do
    (x:xs) <- Just "hello"
    return xs
```

## Monad
### list Monad
1.defination
```
instance Monad [] where
    return x = [x]
    xs >>= f = concat (map f xs)
    fail _ = []
```
```
[1,2] >>= \n -> ['a', 'b'] >>= \ch -> return (n, ch) -- [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
```
[1,2] bind to n, and ['a', 'b'] bind to ch. 

do-block impl
```
listOfTuples :: [(Int, Char)]
listOfTuples = do
    n <- [1,2]
    ch <- ['a', 'b']
    return (n, ch)
```

list comprehension impl
```
[(n, ch) | n <- [1, 2], ch <- ['a', 'b']]
```

```
[x | x <- [1..50], '7' `elem` show x]
```
here `show` is the point.

### `MonadPlus`
1. defination

```
class Monad m => MonadPlus m where
    mzero :: m a
    mplus :: m a -> m a -> m a
```
`mzero` is synonymous to `mempty`, `mplus` to `mappend`.
```
instance MonadPlus [] where
    mzero = []
    mplus = (++)
```

2. `guard`
```
guard' :: (MonadPlus m) => Bool -> m ()
guard' True = return ()
guard' False = mzero
```
not so much useful in normal context, but it's useful in monad chain operations.
```
[1..50] >>= (\x -> guard ('7' `elem` show x) >> return x)
```
so A `guard` basically says: if this boolean is `False` then produce a failure right here,
otherwise make a successful value that has a dummy result of () inside it. all this does is to allow the computation to continue.

example of guard
```
type KnightPos = (Int, Int)

moveKnight :: KnightPos -> [KnightPos]
moveKnight (c, r) = do
    (c', r') <- [(c+2, r-1), (c+2, r+1), (c-2, r-1), (c-2, r+1),
                 (c+1, r-2), (c+1, r+2), (c-1, r-2), (c-1, r+2)]
    guard (c' `elem` [1..8] && r' `elem` [1..8])
    return (c', r')

in3 :: KnightPos -> [KnightPos]
in3 start = do
    first <- moveKnight start
    second <- moveKnight first
    moveKnight second

canReachIn3 :: KnightPos -> KnightPos -> Bool
canReachIn3 start end = end `elem` in3 start
```
I really like this example, this is application regarding `move`

### Writer Monad

#### writer monad primitive impl
```
isBigGang :: Int -> (Bool, String)
isBigGang x = (x > 9, "comp to 9.")

applyLog :: (a, String) -> (a -> (b, String)) -> (b, String)
applyLog (x, log) f = let (y, newLog) = f x in (y, log ++ newLog)
```
should know the `f` produce `(b, String)`, not just a abtrry function working.
how to understand this `f`?
`f x` reprodec `(y, newLog)` then continue works on (y, log ++ newLog)
so `f` type shoulb be `Int -> (Bool, String)`
COOL!!

```
("tobin", "got out ") `applyLog` (\x -> (length x, "applied length. ")) `applyLog` isBigGang
```

#### more closer
```
applyLog' :: (Monoid m) => (a, m) -> (a -> (b, m)) -> (b, m)
applyLog' (x, log) f = let (y, newLog) = f x in (y, log `mappend` newLog)

type Food = String
type Price = Sum Int

addDrink :: Food -> (Food, Price)
addDrink "beans" = ("milk", Sum 25)
addDrink _ = ("beer", Sum 30)
```

#### writer defination
1. defination
```
newtype Writer w a = Writer {runWriter :: (a, w)}

instance (Monad w) => Monad (Writer w) where
    return x = Writer (x, mempty)
    (Writer (x, v)) >>= f = let (Writer (y, v')) = f x Writer (y, v `mappend` v')
```
2. use cases
```
runWriter (return 3 :: Writer String Int)
runWriter (return 3 :: Writer (Sum Int) Int)
```
because Writer has no Show instance, so we use `runWriter` to convert our `Writer` values to normal tuples that can be shown.

```
module WriterMonad where

import Control.Monad.Writer

logNumber :: Int -> Writer [String] Int
logNumber x = writer (x, ["Got number: " ++ show x]) -- here

-- or can use a do-block to do the same thing, and clearly separate the logging from the value
logNumber2 :: Int -> Writer [String] Int
logNumber2 x = do
  tell ["Got number: " ++ show x]
  return x

multWithLog :: Writer [String] Int
multWithLog = do
  a <- logNumber 3
  b <- logNumber 5
  tell ["multiplying " ++ show a ++ " and " ++ show b]
  return (a * b)

main :: IO ()
main = print $ runWriter multWithLog -- (15,["Got number: 3","Got number: 5","multiplying 3 and 5"])
```

another example
```
import qualified Control.Monad as M
import qualified Control.Monad.Trans.Writer.Lazy as W
import Data.Monoid

output :: String -> W.Writer [String] ()
output x = W.tell [x]

gcd' :: Int -> Int -> W.Writer [String] Int
gcd' a b
  | b == 0 = do
    output ("Finished with " ++ show a)
    return a
  | otherwise = do
    output (show a ++ " mod " ++ show b ++ " = " ++ show (a `mod` b))
    gcd' b (a `mod` b)

keepSmall :: Int -> W.Writer [String] Bool
keepSmall x
  | x < 4 = do
    output ("Keeping " ++ show x)
    return True
  | otherwise = do
    output (show x ++ " is too large, throwing it away")
    return False

-- print $ snd $ W.runWriter $ filterM keepSmall [2, 3, 4, 9]
--
powerset :: [a] -> [[a]]
powerset xs = M.filterM (\x -> [True, False]) xs

binSmalls :: Int -> Int -> Either String Int
binSmalls acc x
  | x > 99 = Left ((show x) ++ " is too big")
  | otherwise = Right (acc + x)
```
[cmt] so now why not to summary "what is Writer Monad?"
Writer type constructor is `Writer w a`, `w` is log, but a is value
we should have a function to apply `a` and `tell` will updata `w`.
so application:
```
sumNumber :: Int -> Writer (Sum Int) Int
sumNumber x = do
  tell (Sum x)
  return x

sumNumber 5 >>= \x -> sumNumber x >>= \y -> sumNumber y -- WriterT (Identity (5,
                                                                              Sum
                                                                              {getSum
                                                                              =
                                                                              15}))
logNum :: Int -> Writer [String] Int
logNum x = do
  tell [show x]
  return x

logNum 4 >>= \x -> logNum x >>= \y -> logNum y  -- WriterT (Identity (4, ["4",
                                                                      "4", "4"]))
```

[tricks] print line by line for [String]
```
mapM\_ putStrLn $ snd $ runWriter $ logNum 4 >>= \x -> logNum x
```
### difference list
1. defination

difference list is similar to a normal list, only instead of being a normal
list, it's a function that takes a list and prepends another list to it.
normal list:           [1, 2, 3]
difference list:       \xs -> [1, 2, 3] ++ xs
empty difference list: \xs -> [] ++ xs
and 
two difference list append will be:
```
f `append` g = \xs -> f (g xs)
```
so if `f` is the function : `("dog"++)`
`g` is func: `("meat"++)`
then appended list will be:
```
\xs -> "dog" ++ ("meat" ++ xs)
```

2. encap
```
newtype DiffList a =
  DiffList
    { getDiffList :: [a] -> [a]
    }

toDiffList :: [a] -> DiffList a
toDiffList xs = DiffList (xs ++)

fromDiffList :: DiffList a -> [a]
fromDiffList (DiffList f) = f []
```
how to understan `fromDiffList` `f []`?
diff list is a function, so apply this function to empty list `[]`

become monoid instance
```
instance Monoid (DiffList a) where
  mempty = DiffList (\xs -> [] ++ xs)
  (DiffList f) `mappend` (DiffList g) = DiffList (\xs -> f (g xs))
```
[cmt] in order to be the instance of Monoid, have to impl its basic function.
why become monoid instanc? what is benefit?
could behavior as the same. and we can separately think of them.

```
finalCountDown :: Int -> Writer (DiffList String) ()
finalCountDown 0 = do
  tell (toDiffList ["0"])
finalCountDown x = do
  finalCountDown (x - 1)
  tell (toDiffList [show x])

finalCountDown' :: Int -> Writer [String] ()
finalCountDown' 0 = do
  tell ["0"]
finalCountDown' x = do
  finalCountDown' (x - 1)
  tell [show x]
```
```
mapM\_ putStrLn $ fromDiffList $ snd $ runWriter (finalCount 20000)
mapM\_ putStrLn $ snd $ runWriter (finalCount' 20000)
```

## `State` Monad
### primitive impl 

```
type Stack = [Int]
pop :: Stack -> (Int, Stack)
pop []     = (0, [])
pop (x:xs) = (x, xs)

push :: Int -> Stack -> ((), Stack)
push a xs = ((), a:xs)

stackMainip :: Stack -> (Int, Stack)
stackMainip stack = let
    ((), newStack1) = push 3 stack
    (a, newStack2) = pop newStack1
    in pop newStack2
```
better we impl it as
```
stackMainip = do
    push 3
    a <- pop
    pop
```
### defination
1. stateful computation in `Control.Monad.State`
2. defination
```
newtype State s a = State {runState :: s -> (a, s)}
```
monad instance
```
instance Monad (State s) where
    return x = State $ \s -> (x, s)
    (State h) >>= f = State $ \s -> let (a, newState) = h s
                                        (State g) = f a
                                    in g newState
```
`h` is stateful computation, `>>=` to `f` means we need to unwrap (a, s) from `h` firstly.
in `h` context, we already have a pair (a, s), in order to work continue, we'd better extract this pair from the first context.
so we have `let (a, newState) = h s`. `s` is previous state, `newState` is new produced state.
`State g = f a`, how to understand this? what is type of `f`? `f` is a function, this func works on result `a` and produce a new stateful computation `g`.but how and why?
`g newState` means `g` apply to `newState`, so we get final result, it's a tuple `(a, s)`
`g` is `s-> (a, s)`.
so with bind operator, we kind of glue two stateful computation. so `f` is stateful computation too.

3. state impl
```
pop :: State Stack Int
pop = state $ \(x:xs) -> (x, xs)

push :: Int -> State Stack ()
push a = state $ \xs -> ((), a:xs)

stackMainip :: State Stack Int
stackMainip = do
    push 4
    pop
    pop

```
use case
```
runState stackMainip [3, 4, 6]
```
can't be `runState (stackMainip [3, 4, 6])`, because runState defination, need two parameters.
[cmt] this example, state is a list and result is integer.
so state could be any type.

4. StateMonad
```
get = State $ \s -> (s, s)
put newState = State $ \s -> ((), newState)
(>>=) :: State s a -> (a -> State s b) -> State s b
```

```
stackyStack :: State Stack ()
stackyStack = do
    stackNow <- get
    if stackNow == [1, 2, 3]
       then put [8, 3, 1]
       else put [9, 2, 1]

```
