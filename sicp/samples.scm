;;; https://mitpress.mit.edu/sites/default/files/sicp/index.html

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

(define(list-ref items n)
    (if (= n 0)
        (car items)
        (list-ref (cdr items) (- n 1))))

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
            (len-iter (cdr a) (+ 1 count))))
    (len-i list-c 0))

;;; =========================================================
;;; list append list
(define (append list1 list2)
    (if (null? list1)
        list2
        (cons (car list1) (append (cdr list1) list2))))

(define nil ())
(define (reverse list-c)
    (display list-c)
    (if (null? list-c)
        nil
        (append (reverse (cdr list-c)) (list (car list-c)))))
(reverse one-through-four)

;;; =========================================================
;;; list generic process
(define (scale-list items factor)
    (if (null? items)
        nil
        (cons (* (car items) factor)
            (scale-list (cdr items) factor))))
(scale-list (list 1 2 3 4) 4)

;;; =========================================================
;;; list map, the general map takes a procedure of n arguments, together with
;;; n lists, and applies the procedure to all the first elements
;;; of the lists, all the second elements of the lists and so on,
;;; return a list of the results.
(map + (list 1 2 3) (list 40 50 60) (list 700 800 900))

(map (lambda (x y) (+ x (* 2 y)))
    (list 1 2 3)
    (list 4 5 6))

;;; =========================================================
;;; but we can map, map takes a procedure as argument and list
;;; as another argument, and return a list of the results
;;; produced by applying the procedure to each element in the
;;; list.
(define (map proc items)
   (if (null? items)
        nil
        (cons (proc (car items))
            (map proc (cdr items)))))
(map (lambda (x) (* x x)) (list 1 2 3 4))

(define (scale-list-map items factor)
    (map (lambda (x) (* x factor))
        items))
(scale-list-map (list 2 3 4) 4)

;;; =========================================================
;;; Exc 2.21
(define (square-list items)
    (if (null? items)
        nil
        (cons (* (car items) (car items))
            (square-list (cdr items)))))
(square-list (list 1 2 3))

(define (square-list-2 items)
    (map (lambda (x) (* x x))
        items))
(square-list-2 (list 1 2 3))

;;; =========================================================
;;; Exc 2.23
; (define (for-each-2 proc items) ;; warning, todo
;     (if (null? items)
;         nil
;         (cons (proc (car items))
;             (for-each-2 proc (cdr items)))))

(define (for-each-2 proc items) ;; warning, todo
    (if (null? items)
        nil
        (proc (car items)))
    ((for-each-2 proc (cdr items))))

; (for-each-2 (lambda (x)
;             (newline)
;             (display x))
;             (list 34 23 45))

;;; =========================================================
;;; count leaves
(define (count-leaves items)
    (cond ((null? items) 0)
            ((not (pair? items)) 1)
            (else (+ (count-leaves (car items))
                   (count-leaves (cdr items))))))
(count-leaves one-through-four)

;;; =========================================================
;;; Exc 2.25
(define items (list 1 3 (list 5 7) 9))
(cdr (car (cdr (cdr items))))
(define items (list (list 7)))
(car (car items))
(define items (list 1 (list 2 (list 3 (list 4 (list 5 (list 6 7)))))))

;;; =========================================================
;;; Exc 2.26
(define x (list 1 2 3))
(define y (list 4 5 6))
(append x y) ; (1 2 3 4 5 6)
(cons x y)   ; ((1 2 3) 4 5 6)
(list x y)   ; ((1 2 3) (4 5 6))

;;; =========================================================
;;; Exc 2.27

;;; =========================================================
;;; Exc 2.28

;;; =========================================================
;;; sequences
;;; Q: is it different from lsit? sequence is ordered list
;;; =========================================================

;;; =========================================================
;;; sum of square the odd leaf of a list
(define (sum-odd-square items)
    (cond ((null? items) 0)
        ((not (pair? items))
            (if (odd? items) (+ 0 items) 0))
         (else (+ (sum-odd-square (car items))
               (sum-odd-square (cdr items))))))
(define list-x (cons (list 1 2 3) (list 4)))
(sum-odd-square list-x)

;;; ===========================================================================
;;; IMPORTANT: organize program to make the signal-flow structure in the
;;; procedures, in order to incease the concptual clarity of
;;; the resulting code.
;;;
;;; how?
;;; focus on the signal flow from one stage in the process to the next.
;;; if we represent these signals as lists, then we use list operations to
;;; implement the processing at each of the stages.
;;;
;;; IDEA: (define filter predicate sequence)
;;; means: filter a sequence to select the elements that satify predicate
(define (filter predicate sequence)
    (cond ((null? sequence) nil)
        ((predicate (car sequence))
            (cons (car sequence)
                (filter predicate (cdr sequence))))
        (else (filter predicate (cdr sequence)))))
(filter odd? (list 1 2 3 4))

;;; this is surprising step, I think the following steps maybe the put predicate
;;; into sequence. this will implment signal flow
;;; look at accumulation, this is the pattern for apply the same pattern for the
;;; same type component.
;;; op can be component create, connect, delete
(define (accumulate op initial sequence)
    (if (null? sequence)
        initial
        (op (car sequence)
            (accumulate op initial (cdr sequence)))))
(accumulate + 0 (list 1 2 3 4))
;;; this is the reason why we use more array than list in C, because C list is
;;; a litter complex than C.
(accumulate cons nil (list 1 2 3 4 5))
