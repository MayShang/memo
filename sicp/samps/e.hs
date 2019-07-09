
safe_divide :: Int -> Int -> Either String Int
safe_divide _ 0 = Left "divide by zero"
safe_divide i j = Right (i `div` j)

foo :: Int -> Int -> Int -> Either String Int
foo i j k = case j `safe_divide` k of
                 Left msg -> Left msg
                 Right r -> Right (i + r)

data ArithmeticError = DivideByZero | NotDivisible deriving Show
safe_divide' :: Int -> Int -> Either ArithmeticError Int
safe_divide' _ 0  = Left DivideByZero
safe_divide' i j| i `mod` j /=0 = Left NotDivisible
safe_divide' i j = Right (i `div` j)

divide :: Int -> Int -> Either ArithmeticError Int
divide i j = case i `safe_divide'` j of 
                  Left DivideByZero -> Left DivideByZero
                  Left NotDivisible -> Right (i `div` j)
                  Right k           -> Right k

-- Either nomal impl
-- g i j k = i / k + j / k
g :: Int -> Int -> Int -> Either ArithmeticError Int
g i j k = case i `safe_divide'` k of 
               Left err1 -> Left err1
               Right q1 ->
                   case j `safe_divide'` k of
                        Left err2 -> Left err2
                        Right q2 -> Right (q1 + q2)

-- Either Monad impl
g' :: Int -> Int -> Int -> Either ArithmeticError Int
g' i j k = do
    q1 <- i `safe_divide'` k
    q2 <- j `safe_divide'` k
    return (q1 + q2)

-- Either Monad without do-block
-- we rarely do this in practice (except for very short functions)
-- because do-block is more readable.
-- but you should know how to translation between them.
g'' :: Int -> Int -> Int -> Either ArithmeticError Int
g'' i j k = 
    i `safe_divide'` k >>= \q1 ->
        j `safe_divide'` k >>= \q2 ->
            return (q1 + q2)

