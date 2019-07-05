-- how to define function application inside func
-- 
data IntList = Empty | Cons Int IntList deriving Show
myL = Cons 3 (Cons (-5) Empty)

absA :: IntList -> IntList
absA Empty = Empty
absA (Cons x xs) = Cons (abs x) (absA xs)

mapA :: (Int -> Int) -> IntList -> IntList
mapA _ Empty = Empty
mapA f (Cons x xs) = Cons (f x) (mapA f xs)

-- use case
mapA abs myL
-- cannot use map, map doesn't know IntList, map abs myL
-- but we can use functor

data I a = I a
instance Functor I where
    fmap f I x = I (f x)

-- how to use it?
fmap (+4) (I 5)  -- I 9
fmap (++ "hello ") (I "world ")
fmap ("hello " ++) (I "world ") -- I "hello world"

-- really funny
-- but still not sure how to define a useful data type

-- initution about funcntor
-- give me a function that takes an `a` and return a `b`, and a box with `a`
-- inside it and I'll give you box with a `b` side it. it kind of applies
-- the function to the element inside the box.
-- what does it mean?
-- you give me a function and a box, I'll give a box, this box includes
-- the result of this function.
--
-- so functor keeps shape still, but manipulates its contents.

-- how to make an instance of functor?
-- one:
--   to make what to be a functor?
--   type contructure !!
--   so it's data type we are going to work on.
-- two:
--   this type contructure has to have a kind of `*->*`
--   what does it mean?
--   this type contructure has to take exactly one concrete type as a
--   type parameter.
--   for example, `Maybe` takes one type parameter to produce a concrete one.
--   so `Maybe Int` `Maybe String` working
--   what about two parameter?
--   partially apply the type constructure until it only take one type parameter.
--   so `instance Functor Either where` wrong, you have to write
--   `instance Functor (Either a) where`, but how to work on the first parameter?
--
-- extension:
-- what about IO Functor
-- ```
-- instance Functor IO where
--     fmap f action = do
--         result <- action
--         return (f result)
-- ```
-- mapping sth over an IO action will be an IO action. 

-- ```
-- main = do line <- fmap reverse getLine
--     print $ line
-- ```
--
-- ```
-- main = do line <- fmap (intersperse '-' . reverse . map toUpper) getLine
--     print $ line
-- ```

