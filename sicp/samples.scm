(define (factorial n)
    (fact-iter 1 1 n))
(define (fact-iter product counter max-counter)
    (if (> counter max-counter)
        product
        (fact-iter (* counter product)
                   (+ counter 1)
                   max-counter)))

;(factorial 6)

;;; how to iterative
;;; * values    
;;; * rules

;;; =====================================
;;; sum iterative
;;; ===================================== 
(define (sum_inc a b)
    (sum-iter 1 b b))
(define (sum-iter a result n)
    (if (= a n)
        result
        (sum-iter (+ a 1)
                  (+ a result)
                   n)))

;(sum_inc 1 5)

;;; ===================================== 
(define (sum_dec a b)
    (sum_dec_iter b 0))
(define (sum_dec_iter a result)
    (if (= a 0)
        result
        (sum_dec_iter (- a 1)
                      (+ a result))))

(sum_dec 1 3)

;;; ===================================== 
;;; recursive example from book
;;; ===================================== 
(define (sum term a next b)
    (if (> a b)
        0
        (+ (term a)                     ;; this is every itera result
        (sum term (next a) next b))))   ;; next a is counter

(define (inc n) (+ n 1))
(define (identity x) x)
(define (sum-integers a b)
    (sum identity a inc b))

;;(sum-integers 1 3)

;;; =========================================================
;;;
;;; (define (sum term a next b)
;;;     (define (iter a result)
;;;         (if <??>
;;;             <??>
;;;             <iter <??> <??>)))
;;;     (iter <??> <??>))
;;;
;;; IDEAs: find counter and result value
;;;        and rules of counter and result
;;;        remember every iteration, we can get a result.
;;;        this is iterative advantage. no more space.
;;;        only space(1)
;;;
;;; (define (sum term a next b)
;;;   (define (sum_iter a result))
;;;     (if (= a b)
;;;         result
;;;         (sum_iter (next a)
;;;                   (+ a reult)))
;;;     (sum_iter a b))
;;; =========================================================

(define (sum-i-ver term a next b)
    (define (sum-i-ver-iter a result)
        (if (= a b)
            result
            (sum-i-ver-iter
             (next a)
             (+ (term a) result))))
    (sum-i-ver-iter a b))

(define (sum-xx a b)
    (sum-i-ver identity a inc b))

(sum-xx 1 3)


;;; =========================================================
;;; lambda and let
;;; let expression
;;;
;;; =========================================================
((lambda (x) (+ x 4)) 5)
((lambda (x y z) (+ x y z)) 1 2 3)

(+ (let ((x 3) (y 4))
      (+ x y)) 6)

(define (foo x y)
   (let ((a (+ 1 x))
         (b (- y 2)))
      (+ a x y b)))

(foo 3 7)


;;; =========================================================
;;; best examples
;;; =========================================================

;;; =====================================
;;; 1. how to define a constant
;;; 2. how to define inside another
;;;    procedure
;;;
;;; below procedure doing:
;;; given a data, how to calc the value make
;;; f(?) = data, it use the general
;;; algorithm, really surprised.
;;; NOTE: actually this is model building
;;;       process. we build a frame to
;;;       have a basic algorithm to get our target
;;;       then change different f to improve our
;;;       algorithm.
;;; =====================================
(define tolerance 0.00001)
(define (fixed-point f first-guess)
    (define (close-enough? v1 v2)
        (< (abs (- v1 v2)) tolerance))
    (define (try guess)
        (let ((next (f guess)))
            (if (close-enough? guess next)
                next
                (try next))))
    (try first-guess))

(fixed-point cos 1.0)

;;; this is big and GREAT idea.
;;; this process is the process we used for finding square roots.
;;; both are based on the idea of repeatedly improving a guess until the result satifies some criterion.
(define tol 0.00001)
(define (get-w f target)
    (define (loss-little-enough? v1 v2)
        (< (abs (- v1 v2)) tol))
    (define (calc target)
        (let ((next (f target)))
            (if (loss-little-enough? next target)
                next
                (calc next))))
    (calc target))

(get-w sin 1.0)
(get-w (lambda (y) (+ (sin y) (cos y))) 1.0)


;;; f is procedure, lambda a new procedure, but f can be any function we want to check.
(define (average-dump f)
    (lambda (x) (+ x (f x))))
((average-dump cos) 10)

;;; derivative
(define dx 0.00001)
(define (deriv g)
    (lambda (x) (/ (- (g (+ x dx)) (g x)) dx)))
(define (cube x) (* x x x))
((deriv cube) 5)

;;; represent abstractions explicitly as elements in our programming language,
;;; so that they can be handled just like other computational elements.



;;; =========================================================
;;; best concepts and ideas
;;; 1. evolution of a process, how to this evolution happen?
;;;    procedure can be regarded as pattern for this evolution.
;;;    but still how? we have a algorithm, or say a pattern,
;;;    let process walk through again and again along with the
;;;    updated data. where data from? calc from pattern (algorithm.)
;;; 2. how the abstraction as a technique to cope with complexity.
;;;    how data abstraction build abstraction barriers between
;;;    different parts of a program.
;;; 3. what is glue to combine and form more complex data objects.
;;; 4. rational system, represented by cons, car, cdr
;;; (define made-rat cons)
;;; (define numer car)
;;; (define denom cdr)
;;; =========================================================
(define (make-rat n d) (cons n d))
(define (numer x) (car x))
(define (denom x) (cdr x))
(define (print-rat x)
    (newline)
    (display (numer x))
    (display "/")
    (display (denom x)))

(define one-half (make-rat 1 2))
(print-rat one-half)

(print-rat ((lambda (x y) (make-rat x y)) 4 5))

(define (make-rat n d)
    (let ((g (gcd n d)))
     (cons (/ n d) (/ d g))))

(print-rat ((lambda (x y) (make-rat x y)) 7 5))

;;; =========================================================
;;; what is meant by data? not every arbitrary set of three
;;; procedures can serve as an appropriate basis for the
;;; rational-number implementation.
;;; =========================================================


;;; =========================================================
;;; list
;;; =========================================================
(define one-through-four (list 1 2 3 4))
one-through-four
(cons 5 one-through-four)
(define xxx (cons 6 one-through-four)) ;;
(display xxx)                          ;; wrong
xxx                                    ;; this is list print method

;;; list length
;;; len recursive
(define (len list-c)
    (if (null? list-c)
        0
        (+ 1 (len (cdr list-c)))))
(len one-through-four)

;;; iterative
(define (len-i list-c)
    (define (len-iter a count)
        (if (null? a)
            count
            (len-iter (cdr a) (+ 1 count)))))

;;; list append list
(define (append list1 list2)
    (if (null? list1)
        list2
        (cons (car list1) (append (cdr list1) list2))))
(define nil ())
(define (reverse list-c)
    (if (null? list-c)
        nil
        (list (reverse (cdr list-c)) (car list-c)))) ;; wrong, todo
(reverse one-through-four)
