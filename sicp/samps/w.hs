module WriterMonad where

import Control.Monad.State

-- From http://learnyouahaskell.com/for-a-few-monads-more
-- This example no longer works without tweaking - see
-- http://stackoverflow.com/questions/11684321/how-to-play-with-control-monad-writer-in-haskell
-- just replace the data constructor "Writer" with the function "writer" in the line marked "here"
-- That changed with mtl going from major version 1.* to 2.*, shortly after LYAH and RWH were written
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

sumNumber :: Int -> Writer (Sum Int) Int
sumNumber x = do
  tell (Sum x)
  return x

logNum :: Int -> Writer [String] Int
logNum x = do
  tell [show x]
  return x

newtype DiffList a =
  DiffList
    { getDiffList :: [a] -> [a]
    }

toDiffList :: [a] -> DiffList a
toDiffList xs = DiffList (xs ++)

fromDiffList :: DiffList a -> [a]
fromDiffList (DiffList f) = f []

instance Monoid (DiffList a) where
  mempty = DiffList (\xs -> [] ++ xs)
  (DiffList f) `mappend` (DiffList g) = DiffList (\xs -> f (g xs))

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

type Stack = [Int]

stackyStack :: State Stack ()
stackyStack = do
  stackNow <- get
  put [3, 5, 1]

push :: Int -> State Stack Int
push a = state $ \xs -> (a, a : xs)

pop :: State Stack Int
pop = state $ \(x:xs) -> (x, xs)

x :: State Stack Int
x = do
  push 3
  pop
  pop

main :: IO ()
main = print $ runWriter multWithLog -- (15,["Got number: 3","Got number: 5","multiplying 3 and 5"])
