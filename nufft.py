import cmath
from typing import Union
from typing import Any
from enum import Enum
import time
import math
import pyfftw
import matplotlib.pyplot as mp
import pytest as pytest
from scipy.optimize import curve_fit
import numpy as np
import pandas


class fft_direction(Enum):  # simply defining the direction class as we were told
    FORWARD = 1
    BACKWARD = 2


def dft_1d(vec: list[complex], direction: fft_direction) -> Union[list[complex], bool]:
    """

    :param vec: must be a list or numpy array, containing only complex or real numbers
    if not it raises TypeError

    :param direction: fft_direction, can be only fft_direction.FORWARD or
    fft_direction.BACKWARD, if not it raises ValueError

    :return: returns list[complex] after doing the dft_1d algorithm we read in the project in
    the wanted direction, of course only when vec and direction are fitting
    """

    if isinstance(vec, (list, np.ndarray)) is False:
        # checking if the vec is python or np array
        print("The given input isn't an array")
        return False

    if all(isinstance(i, (complex, int, float)) for i in vec) is False:
        # checking if the elements in the array are numbers
        print("The given input list include non-complex number variables")
        return False
    n: int = len(vec)
    if n == 0:
        print("The given input is empty!")
        return False
    newvec: list[complex] = [0j] * n  # initializing a new vector which later going to be A's or a's
    if direction == fft_direction.FORWARD:  # vec is a's, we want to return A's
        for t in range(0, n):
            newvec[t] = sum(vec[i] * cmath.exp((2 * cmath.pi * 1j * i * t) / n)
                            for i in range(0, n))
        return newvec
        # This was the calculation for the forward
    if direction == fft_direction.BACKWARD:
        for t in range(0, n):
            newvec[t] = sum(vec[i] * cmath.exp((-2 * cmath.pi * 1j * i * t) / n) for i in
                            range(0, n)) / n
            # This was the calculation for the inverse.
        return newvec
    print("Wrong direction!!")
    return False


def dft_1d_matrix(vec: list[complex], direction: fft_direction) -> Union[list[complex], bool]:
    """
    In this function we do the same thing we did in dft_1d matrix but now use matrix
    multiplication

    :param vec: must be a list or numpy array, containing only complex or real numbers
    if not it raises TypeError

    :param direction: fft_direction, can be only fft_direction.FORWARD or
    fft_direction.BACKWARD, if not it raises ValueError

    :return: returns list[complex] after doing the dft_1d algorithm we read in the project in
    the wanted direction, of course only when vec and direction are fitting
    """
    if not isinstance(vec, (list, np.ndarray)):  # checking if the vec input is a python or np array
        print("The given input isn't an array")
        return False
    if not all(isinstance(i, (complex, float, int)) for i in vec):
        # checking if the elements are all numbers
        print("The given input list include non-complex number variables")
        return False
    n: int = len(vec)
    if n == 0:
        print("The given input is empty!!")
    if direction == fft_direction.FORWARD:
        newvec: list[complex] = [0] * n
        mf: list[list[complex]] = [[0] * n for i in range(n)]
        # Initializing the matrix for later use, matrix forward
        for i in range(n):
            for k in range(n):
                mf[i][k] = cmath.exp(2 * cmath.pi * i * k * 1j / n)
                # making the matrix element by element from the formula
            newvec = np.matmul(vec, mf).tolist()
            # making np array to a normal python list so we can see the difference later
        return newvec
    if direction == fft_direction.BACKWARD:
        mb: list[list[complex]] = [[0] * n for i in range(n)]
        # Initializing the matrix for later use, matrix backward
        newvec: list[complex] = [0] * n
        for i in range(n):
            for k in range(n):
                mb[i][k] = cmath.exp(
                    -2 * cmath.pi * i * k * 1j / n) / n
                # making the matrix element by element from the formula
            newvec: list[complex] = np.matmul(vec, mb).tolist()
            # making np array to a normal python list so we can see the difference later
        return newvec
    print("Wrong direction")
    return False


def least_squares(vec: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    finding the b0 and b1 for the least squares: y= b0 +b1x
    :param vec: we get numpy array containing only complex numbers, we don't test
    because it's used on known vec
    :param x: self explained
    :return: returning the b0 and b1 floats which build the least squares line
    """
    # finding the b0 and b1 for the least squares: y= b0 +b1x
    xmean: float = float(np.ravel(np.mean(x)))
    ymean: float = float(np.ravel(np.mean(vec)))
    beta1: float = float(np.ravel(np.sum((x - xmean) * (vec - ymean)) / np.sum((x - xmean) ** 2)))
    beta0: float = ymean - beta1 * xmean
    return beta0, beta1


def plot_dft_timing() -> bool:
    """
    we will compare using plots the timing of both project1 functions with the same input
    """

    n: list[list[complex]] = [[i * 1j for i in range(0, t + 1)] for t in range(0, 100)]
    # Initialized all my samples
    normaltimelistsfor: list[float] = [0] * 100
    # list of all the sampling of times for the normal function forward
    matrixtimelistsfor: list[float] = [0] * 100
    # list of all the sampling of times for the matrix function forward
    normaltimelistsback: list[float] = [0] * 100
    # list of all the sampling of times for the normal function backward
    matrixtimelistsback: list[float] = [0] * 100
    # list of all the sampling of times for the normal function backward
    for i in range(0, 100):  # here we are just simply calculating the running times.
        start: float = time.perf_counter()
        dft_1d(n[i], fft_direction.FORWARD)
        end: float = time.perf_counter()
        normaltimelistsfor[i] = end - start
        start = time.perf_counter()
        dft_1d(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        normaltimelistsback[i] = end - start
        start = time.perf_counter()
        dft_1d_matrix(n[i], fft_direction.FORWARD)
        end = time.perf_counter()
        matrixtimelistsfor[i] = end - start
        start = time.perf_counter()
        dft_1d_matrix(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        matrixtimelistsback[i] = end - start
    x: np.ndarray = np.array([t for t in range(1, 101)])
    # these are the original values of lengths of inputs
    xlog: np.ndarray = np.log(x)  # doing log on all the variables to start the curve fitting
    normalforlog: np.ndarray = np.log(normaltimelistsfor)
    # we are doing all this because log(cn^a)=log(c)+a*log(n)
    normalbacklog: np.ndarray = np.log(normaltimelistsback)
    matrixforlog: np.ndarray = np.log(matrixtimelistsfor)
    matrixbacklog: np.ndarray = np.log(matrixtimelistsback)
    clognf: float  # means the log(c) of the normal function forward direction
    anf: float  # means the a of the normal function forward direction
    clognb: float  # means the log(c) of the normal function backward direction
    anb: float  # means the a of the normal function backward direction
    clogmf: float  # means the log(c) of the matrix function forward direction
    amf: float  # means the a of the matrix function forward direction
    clogmb: float  # means the a of the matrix function backward direction
    amb: float  # means the a of the matrix function backward direction
    clognf, anf = least_squares(normalforlog, xlog)
    # finding the log of c and a fitting for the normal function with forward direction
    clognb, anb = least_squares(normalbacklog, xlog)
    # finding the log of c and a fitting for the normal function with backward direction
    clogmf, amf = least_squares(matrixforlog, xlog)
    # finding the log of c and a fitting for the matrix function with forward direction
    clogmb, amb = least_squares(matrixbacklog, xlog)
    # finding the log of c and a fitting for the matrix function with backward direction
    cnf: float = np.exp(clognf)  # c = e^log(c)
    cnb: float = np.exp(clognb)  # c = e^log(c)
    cmf: float = np.exp(clogmf)  # c = e^log(c)
    cmb: float = np.exp(clogmb)  # c = e^log(c)
    curvefitnf: np.ndarray = cnf * x ** anf  # the curve fitting of normal function forward
    curvefitnb: np.ndarray = cnb * x ** anb  # the curve fitting of normal function backward
    curvefitmf: np.ndarray = cmf * x ** amf  # the curve fitting of matrix function forward
    curvefitmb: np.ndarray = cmb * x ** amb  # the curve fitting of matrix function backward
    mp.figure()
    mp.plot(x, curvefitnf, 'b--', label=f"curve fit Normal: c= {cnf} , a= {anf}")
    mp.plot(x, curvefitmf, 'r--', label=f"curve fit Matrix: c= {cmf} , a= {amf}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Forward functions')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    mp.figure()
    mp.plot(x, curvefitnb, 'b--', label=f"curve fit Normal: c= {cnb} , a= {anb}")
    mp.plot(x, curvefitmb, 'r--', label=f"curve fit Matrix: c= {cmb} , a= {amb}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Backward functions')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    return True
################################################################################### Project 2


def vec_tester(vec) -> bool:
    """
    a simple function which checks if the input vec is okay which means a list and numbers
    can be list[float],list[complex],np.ndarray and that all the elements are fitting
    """
    # a simple function which checks if the input vec is okay which means a list and numbers
    # can be list[float],list[complex],np.ndarray and that all the elements are fitting
    if isinstance(vec, (list, np.ndarray)) is False:  # checking if the vec is python or np array
        raise TypeError("The given input isn't an array")

    if all(isinstance(i, (complex, int, float, np.complex64, np.float64,
                          np.complex128, np.float32, np.int32,
                          np.int64)) for i in vec) is False:
        # checking if the elements in the array are numbers
        raise TypeError("The given input list include non-complex number variables")
    return True


def fft_1d_recursive_radix2_forward(vec: list[complex]) -> Union[list[complex], bool]:
    """
    The function fft_1d_recursive_radix2 sends to this function in the beginning after
    that it calls itself, to do the actual calculations of the
    fft_1d radix2 recursively in the forward direction,

    :param vec: must be a list[complex], list[float] or numpy array with every element
    being a complex or float we don't need to test it before because the function that
    sends to this one sends here only if it is fitting.
    here on every recursion we just check if the length is power of 2,
    and if it isn't- raisng an exception

    :return: if "all is good" we are returning a list[complex], which is a list representing:
    Ak = Ake + Ako * the twiddle factor.
    recursively it goes over all the calculations and in the end returns the wanted list of Ak's

    :raises exception: if the length of the vec isn't even, which means that
    overall the original input isn't power of 2
    """
    n: int = len(vec)

    if n == 2:  # ending the recursion case
        return [vec[0] + vec[1], vec[0]-vec[1]]  # I need to explain why
    if n % 2 == 1:  # doing that to ensure the fact, and to prevent errors next
        raise ValueError("Not possible because we assume that n is a power of 2")
    evec: list[complex] = fft_1d_recursive_radix2_forward(vec[::2])
    # recursing the even part of the vec
    ovec: list[complex] = fft_1d_recursive_radix2_forward(vec[1::2])
    # recursing the odd part of the vec
    newvec: list[complex] = [0j] * n  # putting the result here
    twifactor: complex = cmath.exp(2*cmath.pi*1j / n)  # the twiddle factor of the fft
    for i in range(0, n//2):  # the loop to fit together the recursions results.
        releventtwifactor: complex = pow(twifactor, i)
        # doing the pow to fit the algorithm, e^2pi*1j*k/n
        newvec[i] = evec[i] + ovec[i]*releventtwifactor
        newvec[i+n//2] = evec[i] - ovec[i]*releventtwifactor
        # because e^2pi*1j*(i+n/2)/n=(e^2pi*1j*i/n)*e^pi*j= -(e^2pi*1j*i/n)
    return newvec


def fft_1d_recursive_radix2_backward(vec: list[complex]) -> Union[list[complex], bool]:
    """
   The function fft_1d_recursive_radix2 sends to this function in the beginning after
    that it calls itself, to do the actual calculations of the
    fft_1d radix2 recursively in the inverse\backward direction

    :param vec: must be a list[complex], list[float] or numpy array with every
    element being a complex or floatwe don't need to test it before because the function
    that sends to this one sends here only if it is fitting.
    here on every recursion we just check if the length is power of 2,
    and if it isn't- raisng an exception

    :return:  if "all is good" we are returning a list[complex], which is the
    inverse algorithm of fft_1d radix2 done recursively

    :raises exception: if the length of the vec isn't even, which means that
    overall the original input isn't power of 2
    """
    # doing the same thing, but doing /2 for the a_k's and writing -pi instead of +pi
    # the reason we do /2 instead of /n to fit the ifft is because that every number will
    # get /2 log2(n) times, which is the same as /n on the last vector
    n: int = len(vec)

    if n == 2:  # ending the recursion case
        return [(vec[0] + vec[1])/2, (vec[0]-vec[1])/2]  # I need to explain why
    if n % 2 == 1:  # doing that to ensure the fact, and to prevent errors next
        raise ValueError("Not possible because we assume that n is a power of 2")
    evec: list[complex] = fft_1d_recursive_radix2_backward(vec[::2])
    # recursing the even part of the vec
    ovec: list[complex] = fft_1d_recursive_radix2_backward(vec[1::2])
    # recursing the odd part of the vec
    newvec: list[complex] = [0j] * n  # putting the result here
    twifactor: complex = cmath.exp(-2*cmath.pi*1j/n)  # the twiddle factor of the fft
    for i in range(0, n//2):  # the loop to fit together the recursions results.
        releventtwifactor: complex = pow(twifactor, i)
        # doing the pow to fit the algorithm, e^2pi*1j*k/n
        newvec[i] = (evec[i] + ovec[i]*releventtwifactor)/2
        newvec[i+n//2] = (evec[i] - ovec[i]*releventtwifactor)/2
        # because e^2pi*1j*(i+n/2)/n=(e^2pi*1j*i/n)*e^pi*j= -(e^2pi*1j*i/n)
    return newvec


def fft_1d_recursive_radix2(vec: list[complex], direction: fft_direction) ->\
        Union[list[complex], bool]:
    # testing if input is okay, need to test if it is a vector,if the input is complex and stuff
    """
    Doing the fft_1d algorithm using recursions and the odd and even thing,
    in this specific function we are just doing the base testing on the input of the
    direction and the vector, and then sending for the correct recursion function with the given
    direciton or raising an exception if needed sending to one of the functions:
    fft_1d_recursive_radix2_forward, fft_1d_recursive_radix2_backward

    :param vec: our input vector, we are doing all the operations on it,
    it has to be power of 2, has to be a list or numpy array and all it's
    elements must be float or complex

    :param direction: fft_direction, decides which direction of the fft_1d
    algorithm we are going to use it has only two options:
    fft_direction.FORWARD or fft_direction.BACKWARD for the inverse algorithm

    :return: returns a list[complex] or raises an exception, the value of the
    list[complex] is decided by one of the two functions this one sends to.

    :raises Exception: We raise an exception if the vec is not as we explained earlier,
    and if the direction is not one of the two options.
    """
    vec_tester(vec)
    newvec: list[complex] = [0j] * len(vec)

    if direction == fft_direction.FORWARD:
        if len(vec) == 1:
            return vec
        newvec = fft_1d_recursive_radix2_forward(vec)
    elif direction == fft_direction.BACKWARD:
        if len(vec) == 1:
            return vec
        newvec = fft_1d_recursive_radix2_backward(vec)
    else:
        raise ValueError("Invalid direction!!!")

    return newvec


def bit_reverse(indexlength: int) -> list[int]:
    """
    reversing the bits to be used in the fft_1d_radix2,
    to reverse the order of eooeoeoeoeoeoeoee
    """
    bitnum: int = (indexlength-1).bit_length()
    # how many bits needed for the largest index in the vec
    return [int(f'{i:0{bitnum}b}'[::-1], 2) for i in range(indexlength)]


def fft_1d_radix2(vec: list[complex], direction: fft_direction) ->\
        Union[list[complex], bool]:
    """
    in this function we do fft_1d using iterations and using the odd and even algorithm
    :param vec: list[complex] but can be list[float] or numpy array as well of complex
    or float numbers, the vector input of the fft_1d which we are going to do the
    operations on. the length must be power of 2

    :param direction: fft_direction, decides which direction of the fft_1d algorithm we
    are going to use it has only two options: fft_direction.FORWARD or
    fft_direction.BACKWARD for the inverse algorithm

    :return: we return a list[complex] after doing the wanted direction algorithm on
    the given vec

    :raises Exception: we raise an Exception when either the direction is not
    fft_direction.FORWARD or fft_direction.BACKWARD, or when the input vector isn't
    really a list, or numpy array, or when it has an element which isn't complex or float
    """

    vec_tester(vec)
    n: int = len(vec)
    if n == 0 or not math.log2(n).is_integer():
        raise ValueError("Input length must be a power of 2.")
    if direction == fft_direction.FORWARD:
        sign: int = 1
    elif direction == fft_direction.BACKWARD:
        sign: int = -1
    else:
        raise ValueError("ERROR,INVALID DIRECTION!!!")
    if len(vec) == 1:
        return vec
    newindices: list[int] = bit_reverse(n)
    newvec: list[complex] = [0j] * n
    for i in range(0, n):
        newvec[i] = vec[newindices[i]]  # I hope it works, not really sure
    numiter = int(math.log(n, 2))
    # We are going to make this the number of iterations of going on the evens and odds
    for t in range(0, numiter):
        # we'll go over the size of ajs which we need to calculate each Ak,
        # goes on power of 2
        pairsize: int = 2 ** (t+1)
        # pairsize is the overall size of each pair of even and odd in this iteration
        singlesize: int = pairsize // 2  # The size of each element in each pair
        twifactor: complex = cmath.exp(sign * 2j * cmath.pi / pairsize)
        for pair in range(0, n, pairsize):
            # going over each pair, integer goes on the beginning of each pair
            for k in range(0, singlesize):  # going over each half of the pair
                pks: int = pair+k+singlesize
                relevanttwifactor: complex = (twifactor**k)*newvec[pks]
                newvec[pks] = newvec[pair+k] - relevanttwifactor
                # changing the odd part first
                newvec[pair+k] = newvec[pair+k] + relevanttwifactor
                # changing the even part
    if direction == fft_direction.BACKWARD:
        return [newvec[i]/n for i in range(0, n)]
    return newvec


def cnlogn_model(n: int, c: float):
    """
    just making a "formula" for the model, to show the line as c*n*logn
    """
    return c * n * np.log2(n)


def plot_fft_radix2_timing():
    n: list[list[complex]] = [[i * 1j for i in range(0, 2**t)] for t in range(1, 17)]
    # Initializing all my samples
    recursivetimelistsfor: list[float] = [0] * 16
    # list of all the sampling of times for recursive function forward
    iterativetimelistsfor: list[float] = [0] * 16
    # list of all the sampling of times for iterative function forward
    recursivetimelistsback: list[float] = [0] * 16
    # list of all the sampling of times for recursive function backward
    iterativetimelistsback: list[float] = [0] * 16
    # list of all the sampling of times for iterative function backward
    for i in range(0, 16):
        # here we are just simply calculating the running times.
        start: float = time.perf_counter()
        fft_1d_recursive_radix2(n[i], fft_direction.FORWARD)
        end: float = time.perf_counter()
        recursivetimelistsfor[i] = end - start
        start = time.perf_counter()
        fft_1d_recursive_radix2(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        recursivetimelistsback[i] = end - start
        start = time.perf_counter()
        fft_1d_radix2(n[i], fft_direction.FORWARD)
        end = time.perf_counter()
        iterativetimelistsfor[i] = end - start
        start = time.perf_counter()
        fft_1d_radix2(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        iterativetimelistsback[i] = end - start
    x: list[int] = [2**t for t in range(1, 17)]
    # these are the original values of lengths of inputs
    reccurvefitfor: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, x, recursivetimelistsfor)
    # finding the c and other variables useless to us for the recursive function forward
    reccurvefitback:  tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, x, recursivetimelistsback)
    # finding the c and other variables useless to us for the recursive function backward
    itcurvefitfor:  tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, x, iterativetimelistsfor)
    # finding the c and other variables useless to us for the iterative function forward
    itcurvefitback:  tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, x, iterativetimelistsback)
    # finding the c and other variables useless to us for the iterative function backward
    cfitrecursivefor: float = reccurvefitfor[0][0]  # the c
    cfitrecursiveback: float = reccurvefitback[0][0]  # the c
    cfititerativefor: float = itcurvefitfor[0][0]  # the c
    cfititerativeback: float = itcurvefitback[0][0]  # the c

    mp.figure()
    mp.plot(x, [cnlogn_model(i, cfitrecursivefor) for i in x], 'b--',
            label=f"curve fit Recursive:"f" c= {cfitrecursivefor}")
    mp.plot(x, [cnlogn_model(i, cfititerativefor) for i in x], 'r--',
            label=f"curve fit Iterative:"f" c= {cfititerativefor}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Forward functions')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    mp.figure()
    mp.plot(x, [cnlogn_model(i, cfitrecursiveback) for i in x], 'b--',
            label=f"curve fit Recursive:"f" c= {cfitrecursiveback}")
    mp.plot(x, [cnlogn_model(i, cfititerativeback) for i in x], 'r--',
            label=f"curve fit Iterative:"f" c= {cfititerativeback}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Backward functions')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    return True


################################### Project 3

def fft_1d(vec: Union[np.ndarray, list[complex]], direction: fft_direction) ->\
        Union[list[complex], bool, np.ndarray]:
    """
    Here we are computing the fft_1d using Bluestein's algorithm
    In this function we relied on the fact we found that we can do the convolution
    by doing fft on two lists, multiplying the results and then doing the inverse
    on the multiplied results to get the convolution back done.

    :param vec: the input vector, must be a list or numpy array containing complex
    or float elements only, the length of the vector isn't restricted besides the vec
    must be of length at least 1, if vec not fitting, we will raise an Exception


    :param direction: fft_direction, must be fft_direction.FORWARD for the original
    algorithm or fft_direction.BACKWARD for the inverse, and in other cases -
    we will raise an Exception

    :returns: the function returns the output of the algorithm given  fitting vec and
    direction,  the output is list[complex]

    :raises exception: we raise an exception when the direction is not one of the
    given options, or when the vec is not fitting the only restrictions- which is
    list or numpy array of float or complex elements only

    """

    vec_tester(vec)
    n: int = len(vec)
    if n == 0:
        raise ValueError("Input vector must not be empty.")

    if direction == fft_direction.FORWARD:
        sign: int = 1
        multiplication: float = 1  # we will use it to normalize the results later
    elif direction == fft_direction.BACKWARD:
        sign: int = -1
        multiplication: float = 1 / n
    else:
        raise ValueError("Invalid direction!")

    x: list[complex] = [vec[i] * cmath.exp(sign * cmath.pi * 1j * (i ** 2) / n) for i in range(n)]
    y: list[complex] = [cmath.exp(-sign * cmath.pi * 1j * (i ** 2) / n) for i in range(n)]

    new_n: int = 1  # lengthening the arrays to fit this new length of power of 2
    while new_n < 2 * n - 1:
        new_n *= 2

    stretched_x: list[complex] = x + [0j] * (new_n - n)
    stretched_y: list[complex] = y + [0j] * (new_n - n)

    for i in range(n, new_n):
        stretched_y[i] = 0j
    for i in range(1, n):
        stretched_y[new_n - i] = y[i]  # to make it symmetric

    fft_x: list[complex] = fft_1d_radix2(stretched_x, fft_direction.FORWARD)
    # the stretched x after fft
    fft_y: list[complex] = fft_1d_radix2(stretched_y, fft_direction.FORWARD)
    # the stretched y after fft
    fft_xy: list[complex] = [fft_x[i] * fft_y[i] for i in range(new_n)]
    returned_convolution: list[complex] = fft_1d_radix2(fft_xy, fft_direction.BACKWARD)
    newvec: list[complex] = [0j] * n
    for i in range(n):  # the final touch
        newvec[i] = multiplication * returned_convolution[i] *\
                    cmath.exp(sign * cmath.pi * 1j * (i ** 2) / n)
    return newvec


def plot_fft_1d_timing():
    n: list[list[complex]] = [[i+i*1j for i in range(0, k)] for k in range(1, 201)]
    timelistforward: list[float] = [0] * 200
    timelistbackward: list[float] = [0] * 200
    for i in range(0, 200):
        start: float = time.perf_counter()
        fft_1d(n[i], fft_direction.FORWARD)
        end: float = time.perf_counter()
        timelistforward[i] = end - start
        start = time.perf_counter()
        fft_1d(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        timelistbackward[i] = end - start
    x: list[int] = [t for t in range(1, 201)]  # these are the original values of lengths of inputs
    resultcurvefitfor: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, x, timelistforward)
    cfitfor: float = resultcurvefitfor[0][0]
    resultcurvefitback: tuple[np.ndarray, np.ndarray, Any, Any,Any] =\
        curve_fit(cnlogn_model, x, timelistbackward)
    cfitback: float = resultcurvefitback[0][0]

    mp.figure()
    mp.plot(x, [cnlogn_model(i, cfitfor) for i in x], 'b--', label=
            f"fft_1d curve fit Forward: c= {cfitfor}")
    mp.plot(x, [cnlogn_model(i, cfitback) for i in x], 'r--', label=
            f"fft_1d curve fit Backward: c= {cfitback}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('fft_1d functions      ')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()


def makereal(vec: list[complex]) -> list[float]:
    realvec: list[float] = [float(vec[i].real) for i in range(0, len(vec))]
    return realvec


def fft_1d_real_unpack(vec: list[complex]) -> list[complex]:
    """
    unpacking the result of the forward fft real
    we take the packed output of length n//2 + 1 of the original n size input,
     and return n size output
    """
    n: int = len(vec)
    newvec: list[complex] = [0j] * ((n - 1) * 2)
    newvec[:n] = vec  # the first n elements are the same
    for i in range(1, n - 1):
        newvec[2 * (n - 1) - i] = vec[i].conjugate()
    return newvec


def fft_1d_real(vec: Union[list[float], list[complex], np.ndarray],
                direction: fft_direction) -> Union[list[complex], list[float], bool]:
    """
    Here we do the fft 1d algorithm specifically for vectors with only real elements

    :param vec: when forward direction : must be list or numpy array containing only
     real(float) elements, the length of the vector must be even.
     when backward direction: must be a list or numpy array containing complex elements
     (or real), the length can be anything bigger than 0.
     for any other cases of vec input - we raise an exception

     :param direction: fft_direction, we want to get fft_direction.FORWARD or
     fft_direction.BACKWARD any other case raises an exception. this parameter decides in
      which way we are going to go with the algorithm.

     :returns: when input valid, returns the packed transformed vector, if
      direction is forward then we get a list[complex], if direction is backward then
     we get a list[float]
    """
    if not vec_tester(vec):
        return False
    n: int = len(vec)
    if n == 0:
        raise ValueError("The given input is empty!")
    if direction == fft_direction.FORWARD:
        if n % 2 != 0:
            raise ValueError("The length must be even")
        halfsize = n // 2
        h: list[complex] = [vec[2 * i] + 1j * vec[2 * i + 1] for i in range(halfsize)]
        H: list[complex] = fft_1d(h, fft_direction.FORWARD)
        A: list[complex] = [0] * (halfsize + 1)
        A[0] = sum(vec)
        A[halfsize] = sum(vec[i] * (-1) ** i for i in range(n))
        for k in range(1, halfsize):
            twiddlefactor: complex = np.exp(2j * cmath.pi * k / n)
            Hnkconjugate: complex = H[halfsize - k].conjugate()
            A[k] = (H[k] + Hnkconjugate) / 2 - 1j * (H[k] - Hnkconjugate) * twiddlefactor / 2
        return A
    if direction == fft_direction.BACKWARD:
        # we will notice that in this case our n is actually n/2 +1 of some
        # other N which is the len of the output so we do n-1

        H: list[complex] = [0j] * (n - 1)
        # The H array, which later we will turn into h and then a's,
        for k in range(0, n - 1):  # because of symmetry
            twiddlefactor = np.exp(-2j * cmath.pi * k / (2 * (n-1)))
            Akeven = (vec[k] + vec[n - 1 - k].conjugate()) / 2
            # like in the algorithm, the even of Ak
            Akodd = (vec[k] - vec[n - 1 - k].conjugate()) / 2
            # like in the algorithm, the odd of Ak
            H[k] = Akeven + 1j * Akodd * twiddlefactor
        h: list[complex] = fft_1d(H, fft_direction.BACKWARD)
        # in the inverse we know that h is the inverse of H
        a: list[float] = [0] * (2 * (n-1))
        for i in range(0, n - 1):
            a[2 * i] = h[i].real  # a(2j)
            a[2 * i + 1] = h[i].imag  # a(2j+1)
        return makereal(a)
    raise ValueError("Invalid direction!!!")


def plot_fft_1d_real_timing():
    n: list[list[float]] = [[i for i in range(0, 2 * k)] for k in range(1, 201)]
    timelistforward: list[float] = [0] * 200
    # going to have all the running times of fft_1d forward
    timelistbackward: list[float] = [0] * 200
    # going to have all the running times of fft_1d backward
    realtimelistforward: list[float] = [0] * 200
    # going to have all the running times of fft_1d_real forward
    realtimelistbackward: list[float] = [0] * 200
    # going to have all the running times of fft_1d_real backward
    for i in range(0, 200):
        start: float = time.perf_counter()
        fft_1d(n[i], fft_direction.FORWARD)
        end: float = time.perf_counter()
        timelistforward[i] = end - start
        start = time.perf_counter()
        fft_1d(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        timelistbackward[i] = end - start
        start: float = time.perf_counter()
        fft_1d_real(n[i], fft_direction.FORWARD)
        end: float = time.perf_counter()
        realtimelistforward[i] = end - start
        start = time.perf_counter()
        fft_1d_real(n[i], fft_direction.BACKWARD)
        end = time.perf_counter()
        realtimelistbackward[i] = end - start
    x: list[int] = [2 * t for t in range(1, 201)]
    # these are the original values of lengths of inputs
    resultcurvefitfor: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
            curve_fit(cnlogn_model, x, timelistforward)
    cfitfor: float = resultcurvefitfor[0][0]
    resultcurvefitback: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
            curve_fit(cnlogn_model, x, timelistbackward)
    cfitback: float = resultcurvefitback[0][0]
    realresultcurvefitfor: tuple[np.ndarray, np.ndarray, Any, Any, Any] = \
            curve_fit(cnlogn_model, x, realtimelistforward)
    realcfitfor: float = realresultcurvefitfor[0][0]
    realresultcurvefitback: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
            curve_fit(cnlogn_model, x, realtimelistbackward)
    realcfitback: float = realresultcurvefitback[0][0]

    mp.figure()
    mp.plot(x, [cnlogn_model(i, cfitfor) for i in x], 'b--', label=
                f"fft_1d curve fit Forward: c= {cfitfor}")
    mp.plot(x, [cnlogn_model(i, realcfitfor) for i in x], 'r--', label=
                f"fft_1d_real curve fit Forward:" f" c= {realcfitfor}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Forward functions')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    mp.figure()
    mp.plot(x, [cnlogn_model(i, cfitback) for i in x], 'b--', label=
                f"fft_1d curve fit Backward: c= {cfitback}")
    mp.plot(x, [cnlogn_model(i, realcfitback) for i in x], 'r--', label=
                f"fft_1d_real curve fit Backward: c= "f"{realcfitback}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Backward functions')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()


def fft_1d_fftw(vec: Union[list[complex], np.ndarray], direction: fft_direction) ->\
        list[complex]:
    """
    In this function we are doing fft 1d using the pyfftw library
    since in pyfftw they use a different formula than us we are going to do some changes,
    such as if we want to do the forward direction we will do the inverse of pyfftw and
    multiply it by the legnth. if we want to do the backward direction we will do the
    forward of pyfftw and divide it by the length

    :param vec: the input vector, has to be list or np.ndarray of complex (can be float too)
    numbers, if not- it raises an exception

    :param direction: fft_direction, can be fft_direction.FORWARD or fft_direction.BACKWARD,
    if else it raises an exception.

    :returns: if all inputs are fitting, the function returns a list of complex numbers after
    doing all the calculations as explained.
    """
    vec_tester(vec)
    n: int = len(vec)
    pyfftwinputvec: np.ndarray = np.array(vec, dtype='complex64')
    pyfftwoutputvec: np.ndarray = np.array([0j] * n, dtype='complex64')
    newvec: list[complex] = [0j] * n
    if direction == fft_direction.FORWARD:
        planfft = pyfftw.builders.ifft(pyfftwinputvec)
        pyfftwoutputvec = planfft()
        for i in range(0, n):
            pyfftwoutputvec[i] = pyfftwoutputvec[i] * n
            newvec[i] = pyfftwoutputvec[i]
        return newvec
    if direction == fft_direction.BACKWARD:
        planfft = pyfftw.builders.fft(pyfftwinputvec)
        pyfftwoutputvec = planfft()
        for i in range(0, n):
            pyfftwoutputvec[i] = pyfftwoutputvec[i] / n
            newvec[i] = pyfftwoutputvec[i]
        return newvec
    raise ValueError("Invalid direction!")


def fft_1d_real_fftw(vec: Union[list[complex], list[float], np.ndarray], direction:
                     fft_direction) -> Union[list[float], list[complex]]:
    """
    This function is calling pyfftw real fft function, here no need to normalize since it's similar
    to our given algorithm.

    :param vec: list[float] or np.ndarray of float elements only, when direction is forward,
    and list[complex],list[float] or np.ndarray of float and complex when direction is backward.
    if not fitting it raises an exception

    :param direction: can be only fft_direction.FORWARD and fft_direction.BACKWARD, for any other-
    raises an exception
    """

    vec_tester(vec)
    n: int = len(vec)
    if direction == fft_direction.FORWARD:
        pyfftwinputvec: np.ndarray = np.array(vec, dtype='float64')
        pyfftwoutputvec: np.ndarray = np.array([0j] * (n//2 + 1), dtype='complex64')
        planfftreal = pyfftw.builders.rfft(pyfftwinputvec)
        pyfftwoutputvec = planfftreal()
        return list(pyfftwoutputvec)
    if direction == fft_direction.BACKWARD:
        pyfftwinputvec: np.ndarray = np.array(vec, dtype='complex64')
        pyfftwoutputvec: np.ndarray = np.array([0] * (2 * (n-1)), dtype='float64')
        planfftreal = pyfftw.builders.irfft(pyfftwinputvec)
        pyfftwoutputvec = planfftreal()
        return list(pyfftwoutputvec)
    raise ValueError("Invalid direction!!!")


def compare_to_fftw():
    """
    we are going to compute the relative error and the timings of each of the functions
    I wrote this project The relative error I am going to measure by doing norm on the errors
    for each vector, and dividing by the norm of the output vector of the function we use
    in comparison
    """
    x: list[list[float]] = [[i for i in range(0, 2 * t)] for t in range(1, 513)]
    n: list[int] = [2 * t for t in range(1, 513)]  # the lengthes of the x elements
    relativeerrorfor: np.ndarray = np.array([0] * 512, dtype='float64')
    # forward the relative error of fft_1d with fft_1d_fftw
    relativeerrorrealfor: np.ndarray = np.array([0] * 512, dtype='float64')
    # forward the relative error of fft_1d_real with fft_1d_real_fftw
    relativeerrorback: np.ndarray = np.array([0] * 512, dtype='float64')
    # backward the relative error of fft_1d with fft_1d_fftw
    relativeerrorrealback: np.ndarray = np.array([0] * 512, dtype='float64')
    # backward the relative error of fft_1d_real with fft_1d_real_fftw
    timefor: list[float] = [0] * 512  # running time list of fft_1d forward
    timeback: list[float] = [0] * 512  # running time list of fft_1d backward
    timeforreal: list[float] = [0] * 512  # running time list of fft_1d_real forward
    timebackreal: list[float] = [0] * 512  # running time list of fft_1d_real backward
    timeforfftw: list[float] = [0] * 512  # running time list of fft_1d_fftw forward
    timebackfftw: list[float] = [0] * 512  # running time list of fft_1d_fftw backward
    timeforrealfftw: list[float] = [0] * 512  # running time list of fft_1d_real_fftw forward
    timebackrealfftw: list[float] = [0] * 512  # running time list of fft_1d_real_fftw backward
    start: float = 0   # going to store time soon
    end: float = 0  # going to store time soon
    for i in range(0, 512):
        # at first we will just measure the timing of all the functions, saving the
        # outputs for later in the loop when we want to measure the relative errors
        start = time.perf_counter()
        tempfor: np.ndarray = np.array(fft_1d(x[i], fft_direction.FORWARD), dtype='complex64')
        # We are making them np.ndarray so we can use them later more easily
        end = time.perf_counter()
        timefor[i] = end - start

        start = time.perf_counter()
        tempback: np.ndarray = np.array(fft_1d(x[i], fft_direction.BACKWARD), dtype='complex64')
        end = time.perf_counter()
        timeback[i] = end - start

        start = time.perf_counter()
        temprealfor: np.ndarray = np.array(fft_1d_real(x[i], fft_direction.FORWARD),
                                           dtype='complex64')
        end = time.perf_counter()
        timeforreal[i] = end - start

        start = time.perf_counter()
        temprealback: np.ndarray = np.array(fft_1d_real(x[i], fft_direction.BACKWARD),
                                            dtype='float64')
        end = time.perf_counter()
        timebackreal[i] = end - start

        start = time.perf_counter()
        tempforfftw: np.ndarray = np.array(fft_1d_fftw(x[i], fft_direction.FORWARD),
                                           dtype='complex64')
        end = time.perf_counter()
        timeforfftw[i] = end - start

        start = time.perf_counter()
        tempbackfftw: np.ndarray = np.array(fft_1d_fftw(x[i], fft_direction.BACKWARD),
                                            dtype='complex64')
        end = time.perf_counter()
        timebackfftw[i] = end - start

        start = time.perf_counter()
        tempforrealfftw: np.ndarray = np.array(fft_1d_real_fftw(x[i], fft_direction.FORWARD),
                                               dtype='complex64')
        end = time.perf_counter()
        timeforrealfftw[i] = end - start

        start = time.perf_counter()
        tempbackrealfftw: np.ndarray = np.array(fft_1d_real_fftw(x[i], fft_direction.BACKWARD),
                                                dtype='float64')
        end = time.perf_counter()
        timebackrealfftw[i] = end - start
        # Now we will measure the relative errors of each function, as we explained in the
        # opening of the function
        relativeerrorfor[i] = np.linalg.norm(abs(tempfor - tempforfftw)) / \
            np.linalg.norm(abs(tempforfftw))

        relativeerrorback[i] = np.linalg.norm(abs(tempback - tempbackfftw)) / \
            np.linalg.norm(abs(tempbackfftw))

        relativeerrorrealfor[i] = np.linalg.norm(abs(temprealfor - tempforrealfftw)) / \
            np.linalg.norm(abs(tempforrealfftw))

        relativeerrorrealback[i] = np.linalg.norm(abs(temprealback - tempbackrealfftw)) / \
            np.linalg.norm(abs(tempbackrealfftw))

    # now we will find the curve fitting cnlogn of the timing of each function in each direction
    curvefitfor: tuple[np.ndarray, np.ndarray, Any, Any, Any] = curve_fit(cnlogn_model, n, timefor)
    cfitfor: float = curvefitfor[0][0]  # the c fitting for cnlogn timing

    curvefitback: tuple[np.ndarray, np.ndarray, Any, Any, Any] = curve_fit(cnlogn_model,
                                                                           n, timeback)
    cfitback: float = curvefitback[0][0]  # the c fitting for cnlogn timing

    curvefitforreal: tuple[np.ndarray, np.ndarray, Any, Any, Any] = curve_fit(cnlogn_model,
                                                                              n, timeforreal)
    cfitforreal: float = curvefitforreal[0][0]  # the c fitting for cnlogn timing

    curvefitbackreal: tuple[np.ndarray, np.ndarray, Any, Any, Any] = curve_fit(cnlogn_model,
                                                                               n, timebackreal)
    cfitbackreal: float = curvefitbackreal[0][0]  # the c fitting for cnlogn timing

    curvefitforfftw: tuple[np.ndarray, np.ndarray, Any, Any, Any] = curve_fit(cnlogn_model,
                                                                              n, timeforfftw)
    cfitforfftw: float = curvefitforfftw[0][0]  # the c fitting for cnlogn timing

    curvefitbackfftw: tuple[np.ndarray, np.ndarray, Any, Any, Any] = curve_fit(cnlogn_model,
                                                                               n, timebackfftw)
    cfitbackfftw: float = curvefitbackfftw[0][0]  # the c fitting for cnlogn timing

    curvefitforrealfftw: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, n, timeforrealfftw)
    cfitforrealfftw: float = curvefitforrealfftw[0][0]  # the c fitting for cnlogn timing

    curvefitbackrealfftw: tuple[np.ndarray, np.ndarray, Any, Any, Any] =\
        curve_fit(cnlogn_model, n, timebackrealfftw)
    cfitbackrealfftw: float = curvefitbackrealfftw[0][0]  # the c fitting for cnlogn timing

    # now we are going to start the plotting, first- for the timing of the forward functions
    mp.figure()
    mp.plot(n, [cnlogn_model(i, cfitfor) for i in n], 'b--', label=
            f"curve fit fft_1d :" f" c= {cfitfor}")
    mp.plot(n, [cnlogn_model(i, cfitforreal) for i in n], 'r--', label=
            f"curve fit fft_1d_real :" f" c= {cfitforreal}")
    mp.plot(n, [cnlogn_model(i, cfitforfftw) for i in n], 'darkgreen', label=
            f"curve fit fft_1d_fftw :" f" c= {cfitforfftw}")
    mp.plot(n, [cnlogn_model(i, cfitforrealfftw) for i in n], 'y--', label=
            f"curve fit fft_1d_real_fftw :"f" c= {cfitforrealfftw}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Forward Functions Timing')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    # Now the timing of the backward functions
    mp.figure()
    mp.plot(n, [cnlogn_model(i, cfitback) for i in n], 'b--', label=
            f"curve fit fft_1d :" f" c= {cfitback}")
    mp.plot(n, [cnlogn_model(i, cfitbackreal) for i in n], 'r--', label=
            f"curve fit fft_1d_real :" f" c= {cfitbackreal}")
    mp.plot(n, [cnlogn_model(i, cfitbackfftw) for i in n], 'darkgreen', label=
            f"curve fit fft_1d_fftw :" f" c= {cfitbackfftw}")
    mp.plot(n, [cnlogn_model(i, cfitbackrealfftw) for i in n], 'y--', label=
            f"curve fit fft_1d_real_fftw :"f" c= {cfitbackrealfftw}")
    mp.xlabel('n')
    mp.ylabel('time')
    mp.title('Backward Functions Timing')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.show()
    # And now we will do plotting for the relative errors
    # Forward relative errors, we are going to show them in two different plots because of the
    # difference in sizes
    mp.figure()
    mp.plot(n, relativeerrorfor.real, 'b.', markersize=3, label="fft_1d")

    # fit least squares lines for the points
    leastsquaresfitfftfor = np.poly1d(np.polyfit(n, relativeerrorfor.real, 1))
    mp.plot(n, leastsquaresfitfftfor(n), 'b--')
    mp.xlabel('n')
    mp.ylabel('Relative error')
    mp.title('Forward functions R.E')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.grid(True)

    mp.show()
    mp.figure()
    mp.plot(n, relativeerrorrealfor.real, 'r.', markersize=3, label="fft_1d_real")
    leastsquaresfitfftrealfor = np.poly1d(np.polyfit(n, relativeerrorrealfor.real, 1))
    mp.plot(n, leastsquaresfitfftrealfor(n), 'r--')
    mp.ylim(np.min(relativeerrorrealfor) - 0.01, np.max(relativeerrorrealfor) + 0.01)
    mp.xlabel('n')
    mp.ylabel('Relative error')
    mp.title('Forward functions R.E')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.grid(True)
    mp.show()

    # backward relative errors
    mp.figure()
    mp.plot(n, relativeerrorback, 'b.', markersize=3, label="fft_1d")
    mp.plot(n, relativeerrorrealback, 'r.', markersize=3, label="fft_1d_real")
    # fitting least squares lines for the points
    leastsquaresfitfftback = np.poly1d(np.polyfit(n, relativeerrorback, 1))
    leastsquaresfitfftrealback = np.poly1d(np.polyfit(n, relativeerrorrealback, 1))
    mp.plot(n, leastsquaresfitfftback(n), 'b--')
    mp.plot(n, leastsquaresfitfftrealback(n), 'r--')
    mp.xlabel('n')
    mp.ylabel('Relative error')
    mp.title('Backward functions R.E')
    mp.legend(loc="upper left", fontsize=12, frameon=True)
    mp.grid(True)
    mp.show()

    ################################## Project4


def vec_tester_2d(arr):
    """
     We simply make sure that the arr is 2d rectangular
     and that all it's elements are complex or float
    """
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            vec_tester(arr[i])
    else:
        raise TypeError("This isn't a 2D array")
    n: int = len(arr[0])
    for i in range(len(arr)):
        if n != len(arr[i]):
            raise ValueError("This isn't a rectangle")


def fft_2d(arr: Union[list[list[complex]], np.ndarray], direction: fft_direction)\
        -> list[list[complex]]:
    """

    :param arr: Supposed to be a rectangular array or numpyarray of
     complex numbers, in case that it isn't one of these, raising exception
    :param direction: has to be of fft_direction, can be only fft_direction.FORWARD
    or fft_direction.BACKWARD, any other case raises an exception
    :return: returns a list[list[complex]] of the arr after the transformation
    depending on which direction chosen
    """
    # first we check if the input is good:
    vec_tester_2d(arr)
    if (direction != fft_direction.FORWARD) & (direction != fft_direction.BACKWARD):
        raise TypeError("Invalid direction")
    # initializing
    m: int = len(arr)
    n: int = len(arr[0])
    transformedarr: list[list[complex]] = [[0j for i in range(n)] for t in range(m)]
    # Now we start computing, we remember that we kinda do column fft on fft of the rows

    rowsfft: list[list[complex]] = [fft_1d(vec, direction) for vec in arr]
    for j in range(n):
        column = [rowsfft[i][j] for i in range(m)]
        colsfft = fft_1d(column, direction)
        for i in range(m):
            transformedarr[i][j] = colsfft[i]
    return transformedarr


def cfft_2d(arr: Union[list[list[complex]], np.ndarray], direction: fft_direction):
    """
    This function basically gets a rectangle array of complex and does
    a bit of centering changes and running the function fft_2d
    and recentering again
     :param arr: Supposed to be a rectangular array or numpyarray of
     complex numbers, in case that it isn't one of these, raising exception
    :param direction: has to be of fft_direction, can be only fft_direction.FORWARD
    or fft_direction.BACKWARD, any other case raises an exception
    :return: returns a list[list[complex]] of the arr after the transformation
    depending on which direction chosen
    """
    # check if input is okay
    vec_tester_2d(arr)
    if (direction != fft_direction.FORWARD) & (direction != fft_direction.BACKWARD):
        raise TypeError("Invalid direction")
    # initializing
    m: int = len(arr)
    n: int = len(arr[0])
    uncentered_arr: list[list[complex]] = [[0j for i in range(n)] for t in range(m)]
    # Using the hint to make the "centered" array to the array we can use in Task 1
    for i in range(m):
        for t in range(n):
            uncentered_arr[i][t] = arr[i][t] * ((-1) ** (i+t))
    fftarr: list[list[complex]] = fft_2d(uncentered_arr, direction)  # notice it is uncentered
    # Now we're centering the fftarr back
    for i in range(m):
        for t in range(n):
            fftarr[i][t] = fftarr[i][t] * ((-1) ** (i+t))
    return fftarr


###################################### Project 7
def real_vec_tester(v):
    """
        a simple function which checks if the input vec is okay which means a list and numbers
        can be list[float],np.ndarray and that all the elements are fitting
        """
    # a simple function which checks if the input vec is okay which means a list and numbers
    # can be list[float],list[complex],np.ndarray and that all the elements are fitting
    if isinstance(v, (list, np.ndarray)) is False:  # checking if the vec is python or np array
        raise TypeError("The given input isn't an array")

    if all(isinstance(i, (int, float, np.float64, np.float32,
                          np.int64, np.int32)) for i in v) is False:
        # checking if the elements in the array are numbers
        raise TypeError("The given input list include non-real number variables")
    return True


def nufft_1d_I(alpha: np.ndarray, w: np.ndarray, precision: str) -> np.ndarray:
    """

    Parameters
    ----------
    alpha: an array of N+1 complex elements, N must be even
    w: an array of N+1 float elements between -N/2 and N/2
    precision: can be 'single' or 'double', changing how many numbers after the dot we save.

    Returns
    -------
    an array of N+1 complex elements that were transformed as in Algorithm 1 in the paper.

    """
    # step Init
    # we choose the b, m , q exactly as written in page 1384
    if precision == 'single':
        float_dtype = np.float32
        b: float_dtype = float_dtype(0.5993)
        comptype = np.complex64
        q: int = 10

    elif precision == 'double':
        float_dtype = np.float64
        b: float_dtype = float_dtype(1.5629)
        comptype = np.complex128
        q: int = 28
    else:
        raise ValueError("precision must be 'single' or 'double'")
    m: int = 2
    real_vec_tester(w)
    vec_tester(alpha)

    if len(alpha) != len(w):
        raise TypeError("alpha and w must be of the same length")
    # I need to test the alpha and the w's
    # now we continue with what they wanted us to do in page 1380
    alpha = np.asarray(alpha, dtype=comptype)
    w: np.ndarray = np.array(w, dtype=float_dtype)
    N: int = len(w) - 1
    if N % 2 != 0:
        raise TypeError("The lengthes must be of an odd size")
    for i in w:
        if i > N // 2 or i < -N // 2:
            raise TypeError("the w values must be in the range[-N//2,N//2]")
    mu: np.ndarray = np.zeros(N+1, dtype=int)
    P: np.ndarray = np.zeros((q + 1, N + 1), dtype=float_dtype)
    bpi: float_dtype = 1 / (2 * np.sqrt(b * math.pi))
    for k in range(0, N+1):
        mu[k] = round(m*w[k])
        for j in range(-q//2, q//2+1):
            e1: float_dtype = np.exp((-(m*w[k]-(mu[k]+j))**2 / (4*b)), dtype=float_dtype)
            P[j + q//2, k] = bpi * e1
    e2: np.ndarray = np.zeros(N+1, float_dtype)
    for j in range(-N//2, N//2+1):
        e2[j+N//2] = np.exp(b*(2*math.pi*j/(m*N))**2, dtype=float_dtype)
    # A sanity test just for us, making sure that we got the correct P
    ex1: np.complex128 = 0
    ex2: np.complex128 = 0
    ex3: np.complex128 = 0
    temp: np.float64 = 0
    x = 1
    for k in range(N+1):
        ex1 = np.exp(1j*x*w[k])
        ex2 = np.exp(b*(x/m)**2)
        ex3 = np.sum(P[i+q//2, k]*np.exp(1j*(mu[k]+i)*x/m) for i in range(-q//2, q//2 +1))
        temp = max(temp, np.abs(ex1 - ex2 * ex3))

    # Step 1
    tau: np.ndarray = np.zeros(m*N, dtype=comptype)

    # Now we compute tau, but we notice that we actually
    # Compute tau_hat from what written beneath Equation 85
    for k in range(N + 1):
        for j in range(-q//2, q//2 + 1):
            i = (mu[k] + j) % (m*N)
            tau[i] += P[j + q//2, k] * alpha[k]
    # Preparing a small sanity test just for us, we want to get that temp1 = temp2
    temp1: comptype = 0
    x: int = -1
    for i in range(-m * N // 2, m * N // 2):
        temp1 += tau[i % (m * N)] * np.exp(1j*i*x/((m*N)**2))
    temp2: comptype = 0
    for k in range(N + 1):
        temp3 = 0
        for i in range(-q // 2, q // 2 + 1):
            temp3 += P[i + q // 2, k] * np.exp(1j*(mu[k]+i)*x/((m*N)**2))
        temp2 += (alpha[k] * temp3)
    # Step 2
    # Now we compute T, which is actually T_hat by doing FFT on the tau_hat
    T: np.ndarray = np.zeros(m*N, dtype=comptype)
    T = fft_1d(tau, fft_direction.FORWARD)
    # Step 3
    # Now we compute f, we notice that since we use % again the operation
    # works like we need as if we have T and not T_hat
    f: np.ndarray = np.zeros(N+1, dtype=comptype)
    for j in range(-N // 2, N // 2 + 1):
        f[j + N // 2] = T[j % (m * N)] * e2[j + N // 2]
    return f


def nufft_1d_II(beta: np.ndarray, x: np.ndarray, precision: str) -> np.ndarray:
    """
    Parameters
    ----------
    beta - an array of N+1 complex elements, N must be even
    x - an array of N+1 float elements between -pi and pi
    precision: can be 'single' or 'double', changing how many numbers after the dot we save.

    Returns
    -------
    an array of N+1 complex elements that were transformed as in Algorithm 2 in the paper.

    """
    if precision == 'single':
        float_dtype = np.float32
        b: float_dtype = float_dtype(0.5993)
        comptype = np.complex64
        q: int = 10
    elif precision == 'double':
        float_dtype = np.float64
        b: float_dtype = float_dtype(1.5629)
        comptype = np.complex128
        q: int = 28
    else:
        raise ValueError("precision must be 'single' or 'double'")
    vec_tester(beta)
    real_vec_tester(x)
    if len(x) != len(beta):
        raise TypeError("the lengthes of x and beta must be the same")
    m: int = 2
    N: int = len(beta) - 1
    for i in x:
        if i > np.pi or i < -np.pi:
            raise TypeError("The x values must be in the range[-pi,pi]")
    if N % 2 != 0:
        raise TypeError("The lengthes must be of an odd size")
    beta: np.ndarray = np.array(beta, dtype=comptype)  # just in case turning it into an np.array
    v: np.ndarray = np.zeros(N + 1, dtype=int)
    Q: np.ndarray = np.zeros((N + 1, q + 1), dtype=float_dtype)
    # compute vector v and matrix Q

    for j in range(0, N + 1):

        v[j] = round(x[j] * m * N / (2 * np.pi))

        for k in range(-q//2, q//2 + 1):
            bpi: float_dtype = 1 / (2 * math.sqrt(b * np.pi))
            # a part of the Qjk calculation
            partofpow1: float_dtype = x[j] * m * N / (2 * math.pi) - (v[j] + k)
            Q[j, k + q//2] = bpi * math.exp(-(partofpow1 ** 2)/(4 * b))

    e: np.ndarray = np.zeros(N + 1, dtype=float_dtype)
    for k in range(-N//2, N//2 + 1):
        e[k + N//2] = math.exp(b * (2 * math.pi * k / (m * N)) ** 2)
    # Step 1
    u: np.ndarray = np.zeros(m * N, dtype=comptype)
    u_tilde: np.ndarray = np.zeros_like(u)
    for k in range(-N//2, N//2 + 1):
        u[k + N] = beta[k + N//2] * e[k + N//2]
    # We use u_tilde in order to do the fft correctly later as we learned in equation (85)
    u_tilde[:N] = u[N:]
    u_tilde[N:] = u[:N]


    # Step 2
    # We do fft on u_tilde in order to get the U_tilde
    U_tilde: np.ndarray = np.array(fft_1d(u_tilde, fft_direction.FORWARD))


    # Step 3
    # Instead of creating U from U_tilde we use the % here so it would
    # work in the same way we need.
    g: np.ndarray = np.zeros(N + 1, dtype=comptype)
    for j in range(N + 1):
        for k in range(-q // 2, q // 2 + 1):
            i = (v[j] + k) % (m*N)
            g[j] += Q[j, k + q // 2] * U_tilde[i]
    return g


def nufft_1d_III(gamma: np.ndarray, w: np.ndarray, x: np.ndarray, precision: str)\
        -> np.ndarray:
    """
    Parameters
    ----------
    gamma - an array of N+1 complex elements, N must be even
    x - an array of N+1 float elements between -pi and pi
    w: an array of N+1 float elements between -N/2 and N/2
    precision: can be 'single' or 'double', changing how many numbers after the dot we save.

    Returns
    -------
    an array of N+1 complex elements that were transformed as in Algorithm 3 in the paper.

    """
    # Pre step, checking that we got correct inputs
    if precision == 'single':
        float_dtype = np.float32
        complex_dtype = np.complex64
        b = float_dtype(0.5993)
        q: int = 10
    elif precision == 'double':
        float_dtype = np.float64
        complex_dtype = np.complex128
        b = float_dtype(1.5629)
        q: int = 28
    else:
        raise ValueError("precision must be 'single' or 'double'")
    real_vec_tester(w)
    real_vec_tester(x)
    vec_tester(gamma)
    if len(w) != len(x) or len(w) != len(gamma) or len(gamma) != len(x):
        raise TypeError("the lengthes of the gamma, w and x must be th same")
    m: int = 2
    N: int = len(w) - 1
    for i in w:
        if i > N//2 or i < -N//2:
            raise TypeError("the w values must be in the range[-N//2,N//2]")
    for i in x:
        if i > np.pi or i < -np.pi:
            raise TypeError("The x values must be in the range[-pi,pi]")
    if N % 2 != 0:
        raise TypeError("The lengthes must be of an odd size")
    mmN: int = N * (m**2)
    # Init Step
    # Computing the P matrix
    P: np.ndarray = np.zeros((N + 1, q + 1), dtype=float_dtype)
    mu: np.ndarray = np.zeros(N + 1, dtype=int)
    bpi: float_dtype = 1 / (2 * np.sqrt(b * math.pi))
    for k in range(N + 1):
        mu[k] = round(m * w[k])
        for j in range(-q // 2, q // 2 + 1):
            P[k, j + q // 2] = bpi * np.exp(-((m * w[k] - (mu[k] + j))**2) / (4 * b))
    # Preparing an array of exponentials for later use
    e1: np.ndarray = np.zeros(mmN + 1, dtype=float_dtype)
    for k in range(-mmN // 2, mmN // 2 + 1):
        e1[k + mmN//2] = np.exp(b * (2 * np.pi * k / mmN)**2)

    # Now we do a small test for me, just to check my sanity about calculating P
    ex1: np.complex128 = 0
    ex2: np.complex128 = 0
    ex3: np.complex128 = 0
    temp: np.float64 = 0
    px = 1.5
    for k in range(N+1):
        ex1 = np.exp(1j*px*w[k])
        ex2 = np.exp(b*(px/m)**2)
        ex3 = np.sum(P[k,i+q//2]*np.exp(1j*(mu[k]+i)*px/m) for i in range(-q//2, q//2 + 1))
        temp = max(temp, np.abs(ex1 - ex2 * ex3))

    # Preparing the R matrix in order to use in the computing of h
    R = np.zeros((N + 1, q + 1), dtype=float_dtype)
    eta = np.zeros(N + 1, dtype=int)
    for j in range(N+1):
        eta[j] = round(x[j] * N * m / (2 * np.pi))
        for k in range(-q // 2, q // 2 + 1):
            R[j, k + q // 2] = bpi * np.exp(-((x[j] * N * m / (2 * np.pi) - (eta[j] + k))**2)
                                            / (4 * b))
    # Preparing another array of exponentials for the last scaling

    e2: np.ndarray = np.zeros(N+1, dtype=float_dtype)
    for j in range(N+1):
        e2[j] = np.exp(b * (x[j] / m)**2)
    

        # Step 1- Computing the v vector
    v: np.ndarray = np.zeros(mmN, dtype=complex_dtype)
    for k in range(N + 1):
        for j in range(-q // 2, q // 2 + 1):
            i = (mu[k] + j) % mmN
            v[i] += gamma[k] * P[k, j + q // 2]
            # What we have here is actually v_hat, using
            # the hint we were given just below equation (85) in the paper.

    # Doing another small test for me, checking that I got the correct v's
    temp1: complex_dtype = 0
    for i in range(-mmN // 2, mmN // 2):
        temp1 += v[i % mmN] * np.exp(1j * i * px / ((m*N)**2))
    temp2: complex_dtype = 0
    for k in range(N + 1):
        temp3 = 0
        for i in range(-q // 2, q // 2 + 1):
            temp3 += P[k, i + q // 2] * np.exp(1j * (mu[k] + i) * px / ((m*N)**2))
        temp2 += (gamma[k] * temp3)

    # Step 2-Scale the coefficients using the first exponential array we
    # prepared
    for k in range(-mmN // 2, mmN // 2 + 1):
        v[k % mmN] *= e1[k + mmN//2]

    # Step 3- Doing the fft
    V: np.ndarray = fft_1d(v, fft_direction.FORWARD)

    # Step 4:
    h: np.ndarray = np.zeros(N + 1, dtype=complex_dtype)
    for j in range(N + 1):
        for k in range(-q // 2, q // 2 + 1):
            i = (eta[j] + k) % mmN
            h[j] += V[i] * R[j, k + q // 2]

    # Step 5: Final scaling
    for j in range(N + 1):
        h[j] *= e2[j]
    return h


def measure_error_inf(approx: np.ndarray, exact: np.ndarray, alpha: np.ndarray) -> np.float64:
    return np.max(np.abs(approx - exact)) / np.sum(np.abs(alpha))


def measure_error_2(approx: np.ndarray, exact: np.ndarray) -> np.float64:
    return np.sqrt(np.sum(np.abs(approx - exact) ** 2) / np.sum(np.abs(exact) ** 2))


def naive_nufft1d_I(alpha: np.ndarray, w: np.ndarray, precision: str) -> np.ndarray:
    """
    doing Algorithm 1 in the direct and less efficient way

    """
    N: int = len(w) - 1
    if precision == 'single':
        result: np.ndarray = np.zeros(N + 1, dtype=np.complex64)
    elif precision == 'double':
        result: np.ndarray = np.zeros(N + 1, dtype=np.complex128)
    else:
        raise ValueError("precision must be single or double!!")
    for j in range(-N//2, N//2 + 1):
        result[j + N//2] = sum(alpha[k] * cmath.exp(2j * cmath.pi * w[k] * j/N) for k in range(N+1))
    return result


def naive_nufft1d_II(beta: np.ndarray, x: np.ndarray, precision: str) -> np.ndarray:
    """
    doing Algorithm 2 in the direct and less efficient way
    """
    N: int = len(x) - 1
    if precision == 'single':
        result: np.ndarray = np.zeros(N + 1, dtype=np.complex64)
    elif precision == 'double':
        result: np.ndarray = np.zeros(N + 1, dtype=np.complex128)
    else:
        raise ValueError("precision must be single or double!!")

    for j in range(0, N+1):
        result[j] = sum(beta[k+N//2] * np.exp(1j * k * x[j]) for k in range(-N//2, N//2 + 1))
    return result


def naive_nufft1d_III(gamma: np.ndarray, w: np.ndarray, x: np.ndarray,
                      precision: str) -> np.ndarray:
    """
    Doing Algorithm 3 in the direct and less efficient way
    """
    N: int = len(x) - 1
    if precision == 'single':
        result: np.ndarray = np.zeros(N + 1, dtype=np.complex64)
    elif precision == 'double':
        result: np.ndarray = np.zeros(N + 1, dtype=np.complex128)
    else:
        raise ValueError("precision must be single or double!!")

    for j in range(0, N+1):
        result[j] = sum(gamma[k] * np.exp(1j*w[k]*x[j]) for k in range(0, N+1))
    return result


def relative_l2_error(a: np.ndarray, b: np.ndarray) -> np.float64:
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def relative_inf_error(a: np.ndarray, b: np.ndarray) -> np.float64:
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def test_simple_nufft_1d_I_accuracy():
    N: int = 64
    w: np.ndarray = np.arange(-N // 2, N // 2 + 1)  # symmetric frequency domain
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    alpha: np.ndarray = real_part + 1j * imag_part

    f_nufft = nufft_1d_I(alpha, w, precision='single')

    f_direct = naive_nufft1d_I(alpha, w, precision='single')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, alpha)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_nufft_1d_I_accuracy():
    N: int = 64
    w: np.ndarray = np.random.normal(loc=0, scale=N//4, size=N+1)
    for i in range(N+1):
        sign = 1
        if w[i] < 0:
            sign = -1
        w[i] = (w[i] % (N//2)) * sign
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    alpha: np.ndarray = real_part + 1j * imag_part

    f_nufft = nufft_1d_I(alpha, w, precision='single')

    f_direct = naive_nufft1d_I(alpha, w, precision='single')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, alpha)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_nufft_1d_I_accuracy2():
    N: int = 128
    w: np.ndarray = np.random.uniform(-N/2, N/2, N+1)
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    alpha: np.ndarray = real_part + 1j * imag_part

    f_nufft = nufft_1d_I(alpha, w, precision='double')

    f_direct = naive_nufft1d_I(alpha, w, precision='double')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, alpha)

    assert e2 < 1e-10
    assert einf < 1e-10


def test_nufft_1d_I_accuracy3():
    N: int = 256
    w: np.ndarray = np.random.uniform(-N/2, N/2, N+1)
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    alpha: np.ndarray = real_part + 1j * imag_part

    f_nufft = nufft_1d_I(alpha, w, precision='single')

    f_direct = naive_nufft1d_I(alpha, w, precision='single')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, alpha)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_simple_nufft_1d_II_accuracy():
    N: int = 64
    x: np.ndarray = np.ones(N+1, dtype=np.float64)

    beta: np.ndarray = np.zeros(N+1, np.complex64)
    beta[0] = 1 + 1j

    f_nufft = nufft_1d_II(beta, x, precision='single')

    f_direct = naive_nufft1d_II(beta, x, precision='single')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, beta)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_nufft_1d_II_accuracy():
    N: int = 64
    x: np.ndarray = np.random.uniform(-math.pi, math.pi, N+1)
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    beta: np.complex128 = real_part + 1j * imag_part

    f_nufft = nufft_1d_II(beta, x, precision='single')

    f_direct = naive_nufft1d_II(beta, x, precision='single')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, beta)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_nufft_1d_II_accuracy2():
    N: int = 128
    x: np.ndarray = np.random.uniform(-math.pi, math.pi, N+1)
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    beta: np.complex128 = real_part + 1j * imag_part

    f_nufft = nufft_1d_II(beta, x, precision='double')

    f_direct = naive_nufft1d_II(beta, x, precision='double')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, beta)

    assert e2 < 1e-10
    assert einf < 1e-10


def test_nufft_1d_II_accuracy3():
    N: int = 256
    x: np.ndarray = np.random.uniform(-math.pi, math.pi, N+1)
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    beta: np.complex128 = real_part + 1j * imag_part

    f_nufft = nufft_1d_II(beta, x, precision='single')

    f_direct = naive_nufft1d_II(beta, x, precision='single')

    # Error analysis
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, beta)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_nufft_1d_III_accuracy():
    N: int = 64
    x: np.ndarray = np.random.uniform(-math.pi, math.pi, N+1)
    w: np.ndarray = np.random.normal(loc=0, scale=N // 4, size=N + 1)
    for i in range(N + 1):
        sign = 1
        if w[i] < 0:
            sign = -1
        w[i] = (w[i] % (N // 2)) * sign
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    gamma: np.ndarray = real_part + 1j * imag_part
    f_nufft = nufft_1d_III(gamma, w, x, precision='single')
    f_direct = naive_nufft1d_III(gamma, w, x, precision='single')
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft,f_direct, gamma)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_nufft_1d_III_accuracy2():
    N: int = 128
    x: np.ndarray = np.random.uniform(-math.pi, math.pi, N+1)
    w: np.ndarray = np.random.uniform(-N/2, N/2, N+1)  # symmetric frequency domain
    real_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N+1)  # random floats in [0, 1)
    gamma: np.ndarray = real_part + 1j * imag_part
    f_nufft = nufft_1d_III(gamma, w, x, precision='double')
    f_direct = naive_nufft1d_III(gamma, w, x, precision='double')
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft,f_direct, gamma)

    assert e2 < 1e-10
    assert einf < 1e-10


def test_nufft_1d_III_accuracy3():
    N: int = 256
    x: np.ndarray = np.random.uniform(-math.pi, math.pi, N + 1)
    w: np.ndarray = np.random.uniform(-N / 2, N / 2, N + 1)  # symmetric frequency domain
    real_part: np.ndarray = np.random.rand(N + 1)  # random floats in [0, 1)
    imag_part: np.ndarray = np.random.rand(N + 1)  # random floats in [0, 1)
    gamma: np.ndarray = real_part + 1j * imag_part
    f_nufft = nufft_1d_III(gamma, w, x, precision='single')
    f_direct = naive_nufft1d_III(gamma, w, x, precision='single')
    e2: np.float64 = measure_error_2(f_nufft, f_direct)
    einf: np.float64 = measure_error_inf(f_nufft, f_direct, gamma)

    assert e2 < 1e-4
    assert einf < 1e-4



def test_simplenufft_1d_III_accuracy():
    N: int = 64
    x: np.ndarray = np.zeros(N+1, dtype=np.float64)  # Make it random
    x[0] = 1

    w: np.ndarray = np.arange(-N // 2, N // 2 + 1)

    gamma: np.ndarray = np.zeros(N+1, dtype=np.complex128)
    gamma[0] = 1
    f_nufft: np.ndarray = nufft_1d_III(gamma, w, x, precision='single')
    f_direct: np.ndarray = naive_nufft1d_III(gamma, w, x, precision='single')
    e2 = measure_error_2(f_nufft, f_direct)
    einf = measure_error_inf(f_nufft,f_direct, gamma)

    assert e2 < 1e-4
    assert einf < 1e-4


def test_wronginput_nufft1():
    N: int = 64
    rightalpha: np.ndarray = np.ones(N+1, dtype=np.complex128) * 1j
    wrongalpha: list = ["LOL", 1, [1, 1]]
    rightw: np.ndarray = np.ones(N+1, dtype=np.float64)
    wrongw: list = ["Lol", 1, [1, 1], rightalpha, rightw * N]
    wrongprecision: list = [1, "LOL"]
    for alpha in wrongalpha:
        try:
            nufft_1d_I(alpha, rightw, "single")
            assert False
        except TypeError:
            pass
    for w in wrongw:
        try:
            nufft_1d_I(rightalpha, w, "single")
            assert False
        except TypeError:
            pass
    for p in wrongprecision:
        try:
            nufft_1d_I(rightalpha, rightw, p)
            assert False
        except ValueError:
            pass
    assert True


def test_wronginput_nufft2():
    N: int = 64
    rightbeta: np.ndarray = np.ones(N+1, dtype=np.complex128) * 1j
    wrongbeta: list = ["LOL", 1, [1, 1]]
    rightx: np.ndarray = np.ones(N + 1, dtype=np.float64)
    wrongx: list = ["Lol", 1, [1, 1], rightbeta, rightx * N]
    wrongprecision: list = [1, "LOL"]
    for beta in wrongbeta:
        try:
            nufft_1d_II(beta, rightx, "single")
            assert False
        except TypeError:
            pass
    for x in wrongx:
        try:
            nufft_1d_II(rightbeta, x, "single")
            assert False
        except TypeError:
            pass
    for p in wrongprecision:
        try:
            nufft_1d_II(rightbeta, rightx, p)
            assert False
        except ValueError:
            pass
    assert True


def test_wronginput_nufft3():
    N: int = 64
    rightgamma: np.ndarray = np.ones(N+1, dtype=np.complex128) * 1j
    wronggamma: list = ["LOL", 1, [1, 1]]
    rightw: np.ndarray = np.ones(N+1, dtype=np.float64)
    wrongw: list = ["Lol", 1, [1, 1], rightgamma, rightw * N]
    rightx: np.ndarray = np.ones(N + 1, dtype=np.float64)
    wrongx: list = ["Lol", 1, [1, 1], rightgamma, rightw * N]
    wrongprecision: list = [1, "LOL"]
    for alpha in wronggamma:
        try:
            nufft_1d_III(alpha, rightw, rightx, "single")
            assert False
        except TypeError:
            pass
    for w in wrongw:
        try:
            nufft_1d_III(rightgamma, w, rightx, "single")
            assert False
        except TypeError:
            pass
    for p in wrongprecision:
        try:
            nufft_1d_III(rightgamma, rightw, rightx, p)
            assert False
        except ValueError:
            pass
    for x in wrongx:
        try:
            nufft_1d_III(rightgamma, rightw, x, "single")
            assert False
        except TypeError:
            pass
    assert True





##########################################################################################


def test_inversability4():
    a: list[list[complex]] = [[1, 2, 3, 4, 5], [1 + 1j, 2 + 1j, 3 + 1j, 4 + 1j, 5 + 1j],
                              [1 + 2j, 2 + 2j, 3 + 3j, 4.5j, 1j]]

    fftfor: list[list[complex]] = fft_2d(a, fft_direction.FORWARD)
    cfftfor: list[list[complex]] = cfft_2d(a, fft_direction.FORWARD)

    fftforback: list[list[complex]] = fft_2d(fftfor, fft_direction.BACKWARD)
    cfftforback: list[list[complex]] = cfft_2d(cfftfor, fft_direction.BACKWARD)

    np.testing.assert_allclose(fftforback, a, atol=1e-8, rtol=0)
    np.testing.assert_allclose(cfftforback, a, atol=1e-8, rtol=0)


def test_wrongdirection4():

    # testing if it gets the direction right and returns errors to user's wrong directions
    vec: list[list[complex]] = [[0], [0], [0]]
    notdirections = ["hello", 3, [fft_direction.FORWARD, fft_direction.BACKWARD]]
    for direction in notdirections:
        try:
            fft_2d(vec, direction)
            assert False
        except TypeError:
            pass
    for direction in notdirections:
        try:
            cfft_2d(vec, direction)
            assert False
        except TypeError:
            pass


def test_badvec4():
    """
    testing bad cases of vectors, like strings, 1d arrays, ints, and 2d arrays which aren't
    rectangle
    we want the functions to raise exceptions
    """
    badvecs = [1, [1], "hello", [1, [1]]]
    anotherbadvec = [[1], [1, 2]]
    for vec in badvecs:
        try:
            fft_2d(vec, fft_direction.FORWARD)
            assert False
        except TypeError:
            pass
        try:
            cfft_2d(vec, fft_direction.FORWARD)
            assert False
        except TypeError:
            pass
        try:
            fft_2d(anotherbadvec, fft_direction.FORWARD)
            assert False
        except ValueError:
            pass
        try:
            cfft_2d(anotherbadvec, fft_direction.FORWARD)
            assert False
        except ValueError:
            pass
    assert True


def test_overall4():
    """
    in this test we don't check cfft_2d because we will have to do the same changes on the
    fft_2d of numpy as we did in cfft_2d, and it loses it's point
    """
    a: list[list[complex]] = \
        [[0.37454012+0.02058449j, 0.95071431+0.96990985j,
          0.73199394+0.83244264j, 0.59865848+0.21233911j, 1],
            [2, 0.15601864+0.18182497j, 0.15599452+0.18340451j,
             0.05808361+0.30424224j, 0.86617615+0.52475643j],
            [3, 0.60111501+0.43194502j, 0.70807258+0.29122914j, 0.02058449+0.61185289j,
             0.96990985+0.13949386j], [4, 0.83244264+0.29214465j, 0.21233911+0.36636184j,
                                       0.18182497+0.45606998j, 0.18340451+0.78517596j]]
    ourfftfor: list[list[complex]] = fft_2d(a, fft_direction.FORWARD)
    numpyfftfor: list[list[complex]] = np.fft.ifft2(a) * 20
    # we are normalizing and doing opposite directions so the algorithms will fit

    ourfftback: list[list[complex]] = fft_2d(a, fft_direction.BACKWARD)
    numpyfftback: list[list[complex]] = np.fft.fft2(a) / 20

    np.testing.assert_allclose(ourfftfor, numpyfftfor, atol=1e-8, rtol=0)
    np.testing.assert_allclose(ourfftback, numpyfftback,  atol=1e-8, rtol=0)


def test_lotsofnumbers4():
    bigm = [[i + t * 1j for i in range(0, 6)] for t in range(0, 100)]
    bign = [[i + t * 1j for i in range(0, 100)] for t in range(0, 6)]
    bignm = [[i + t * 1j for i in range(0, 100)] for t in range(0, 100)]

    ourfftbigmfor = fft_2d(bigm, fft_direction.FORWARD)
    ourfftbigmback = fft_2d(bigm, fft_direction.BACKWARD)

    ourfftbignfor = fft_2d(bign, fft_direction.FORWARD)
    ourfftbignback = fft_2d(bign, fft_direction.BACKWARD)

    ourfftbignmfor = fft_2d(bignm, fft_direction.FORWARD)
    ourfftbignmback = fft_2d(bignm, fft_direction.BACKWARD)

    numpyfftbigmfor = np.fft.ifft2(bigm) * 600
    numpyfftbigmback = np.fft.fft2(bigm) / 600
    numpyfftbignfor = np.fft.ifft2(bign) * 600
    numpyfftbignback = np.fft.fft2(bign) / 600
    numpyfftbignmfor = np.fft.ifft2(bignm) * 10000
    numpyfftbignmback = np.fft.fft2(bignm) / 10000

    np.testing.assert_allclose(ourfftbigmfor, numpyfftbigmfor,  atol=1e-8, rtol=0)
    np.testing.assert_allclose(ourfftbigmback, numpyfftbigmback,  atol=1e-8, rtol=0)
    np.testing.assert_allclose(ourfftbignfor, numpyfftbignfor,  atol=1e-8, rtol=0)
    np.testing.assert_allclose(ourfftbignback, numpyfftbignback,  atol=1e-8, rtol=0)
    np.testing.assert_allclose(ourfftbignmfor, numpyfftbignmfor,  atol=1e-8, rtol=0)
    np.testing.assert_allclose(ourfftbignmback, numpyfftbignmback,  atol=1e-8, rtol=0)


def test_inversability3():
    a: list[float] = [1, 2, 3, 4, 5, 6, 7, 8]

    # Complex FFT forward and backward
    A: list[complex] = fft_1d(a, fft_direction.FORWARD)
    back: list[complex] = fft_1d(A, fft_direction.BACKWARD)

    # Real FFT forward and backward
    Areal: list[complex] = fft_1d_real(a, fft_direction.FORWARD)
    back_real: list[complex] = fft_1d_real(Areal, fft_direction.BACKWARD)
    # Assert complex inverse returns original
    np.testing.assert_allclose(back, a, atol=1e-8, rtol=0)
    # Assert real inverse returns original
    np.testing.assert_allclose(back_real, a, atol=1e-8, rtol=0)


def test_overall3():
    a: list[complex] = [4.32, -1, 1.47, -6.32]
    n: int = len(a)

    # what we got from our fft_1d
    fftfor = fft_1d(a, fft_direction.FORWARD)
    fftback = fft_1d(a, fft_direction.BACKWARD)

    # what we wanted to get from our functions, we will normalize the result
    # to fit the formula we use
    wantedback = np.fft.fft(a) / n
    wantedfor = np.fft.ifft(a) * n

    np.testing.assert_allclose(fftfor, wantedfor, atol=1e-8, rtol=0)
    np.testing.assert_allclose(fftback, wantedback, atol=1e-8, rtol=0)


def test_overallreal3():  # testing another case for the real functions
    even: list[float] = [4.3, 5.4, -2, 3.3, 6.7, 0]
    odd: list[float] = [0, 0, 3, 4, 5]

    ourfftrealfor: list[complex] = fft_1d_real(even, fft_direction.FORWARD)
    ourfftrealback: list[float] = fft_1d_real(odd, fft_direction.BACKWARD)
    ourfftwrealfor: list[complex] = fft_1d_real_fftw(even, fft_direction.FORWARD)
    ourfftwrealback: list[float] = fft_1d_real_fftw(odd, fft_direction.BACKWARD)

    numpyfftrealfor: list[complex] = np.fft.rfft(even)
    numpyfftrealback: list[float] = np.fft.irfft(odd)
    try:
        np.testing.assert_allclose(ourfftwrealfor[:4], numpyfftrealfor[:4], rtol=0, atol=1e-4)
        np.testing.assert_allclose(ourfftwrealback[:8], numpyfftrealback[:8], rtol=0, atol=1e-4)
    except AssertionError:
        assert False
    else:
        assert True

    try:
        np.testing.assert_allclose(ourfftrealfor[:4], numpyfftrealfor[:4], rtol=0, atol=1e-4)
    except AssertionError:
        try:
            np.testing.assert_allclose([x.conjugate() for x in ourfftrealfor[:4]],
                                       numpyfftrealfor[:4], rtol=0, atol=1e-4)

        except AssertionError:
            assert False

    try:
        np.testing.assert_allclose(ourfftrealback[:8], numpyfftrealback[:8], rtol=0, atol=1e-4)
    except AssertionError:
        assert False
    else:
        assert True


def test_lotofnumbers3():
    # Long list to test FFT implementation
    a: list[float] = [34.2, -12.4, 56.1, -99.9, 87.2,
                      23.5, -45.3, 19.8, 90.1, -3.4,
                      12.6, -67.2, 45.1, 23.4, -90.4,
                      78.9, -34.7, 11.1, 65.3, -76.9,
                      90.5, -88.6, 34.5, 12.2, -55.4,
                      100.1, -23.6, 56.7, 45.9, -12.3,
                      76.4, -34.8, 19.7, 90.8, -77.2,
                      12.5, -56.3, 34.1, 89.7, -23.5,
                      100.2, -56.8, 12.4, 78.5, -99.1,
                      23.6, -67.4, 12.1, 56.2, -45.7,
                      90.3, -78.6, 34.2, 12.3, -56.5,
                      45.1, -67.2, 100.6, -34.3, 12.4,
                      90.1, -99.2, 56.4, 78.6, -34.5,
                      11.2, -45.8, 12.9, 67.5, -56.6,
                      34.7, -78.9, 100.9, -23.8, 45.2,
                      90.7, -34.6, 12.5, -56.7, 78.3,
                      -99.8, 34.9, -12.6, 56.8, -78.4,
                      90.2, -23.9, 67.8, -45.3, 12.7,
                      34.8, -99.7, 56.9, 78.1, -100.4,
                      -12.8, 91.4, 66.5, -45.6,
                      14.9, -87.3, 53.7, -26.4,
                      77.1, -68.9, 39.2, 100.5,
                      -55.1, 88.6, -12.2, 67.4,
                      23.1, -44.7, 89.3, -98.6,
                      35.4, -100.0, 12.2, 56.7,
                      -66.6, 45.3, 91.6, -32.7,
                      100.3, 18.8, -78.2, -92.13, 92.13]

    n = len(a)
    b = fft_1d(a, fft_direction.FORWARD)
    d = fft_1d(a, fft_direction.BACKWARD)

    f = np.fft.fft(a) / n
    i = np.fft.ifft(a) * n

    np.testing.assert_allclose(b, i, rtol=0, atol=1e-4)
    np.testing.assert_allclose(d, f, rtol=0, atol=1e-4)


def test_wrongdirection3():
    # testing if it gets the direction right and returns errors to user's wrong directions
    vec: list[float] = [0, 0, 0]
    notdirections = ["hello", 3, [fft_direction.FORWARD, fft_direction.BACKWARD]]
    for direction in notdirections:
        try:
            fft_1d(vec, direction)
            assert False
        except ValueError:
            pass
    for direction in notdirections:
        try:
            fft_1d_real(vec, direction)
            assert False
        except ValueError:
            pass
    for direction in notdirections:
        try:
            fft_1d_fftw(vec, direction)
            assert False
        except ValueError:
            pass
    for direction in notdirections:
        try:
            fft_1d_real_fftw(vec, direction)
            assert False
        except ValueError:
            pass
    assert True


def test_notvec3():
    """
    checking if the input checker for the vec is correct in each function from
    project 3
    """
    notarray: int = 55214142123146
    stringarray: list[object] = [5j, 3, "hello"]
    tuplenotarray: tuple[complex, complex, complex] = (5j, 3j, 10j)
    notvecs = [notarray, stringarray, tuplenotarray]  # for easier use later with iteration
    for notvec in notvecs:
        try:
            fft_1d(notvec, fft_direction.FORWARD)
            assert False  # If no error happens, the test should fail
        except TypeError:
            pass  # If an error happens, that's what we want

    for notvec in notvecs:
        try:
            fft_1d_real(notvec, fft_direction.FORWARD)
            assert False
        except TypeError:
            pass
    for notvec in notvecs:
        try:
            fft_1d_fftw(notvec, fft_direction.FORWARD)
            assert False
        except TypeError:
            pass
    for notvec in notvecs:
        try:
            fft_1d_real_fftw(notvec, fft_direction.FORWARD)
            assert False
        except TypeError:
            pass
    assert True


def test_oddvec3():
    """
    we will test what happens in the forward of fft_1d_real
    what happens when we enter a vec with odd number of elements.
    we don't check for the others since they aren't restricted
    """
    # we will test what happens in the function fft_1d, and the forward of fft_1d_real
    # what happens when we enter a vec with odd number of elements.
    # we don't check for
    oddvecs: list[list[complex]] = [[0, 0, 0], [1, 2, 3, 4, 5], [1j, 4j, 0j, 4.32, -3, 0, 0, 0, 0]]
    for vec in oddvecs:
        try:
            fft_1d_real(vec, fft_direction.FORWARD)
            assert False
        except ValueError:
            pass
    assert True


def test_inversability2():
    """
    checking if it kind of returns to itself after going both ways
    """
    a: list[complex] = [1j, 2j, 3j, 4j]
    afor: list[complex] = fft_1d_recursive_radix2(a, fft_direction.FORWARD)
    forback: list[complex] = fft_1d_recursive_radix2(afor, fft_direction.BACKWARD)
    aforiter: list[complex] = fft_1d_radix2(a, fft_direction.FORWARD)
    forbackiter: list[complex] = fft_1d_radix2(aforiter, fft_direction.BACKWARD)
    np.testing.assert_allclose(forback, a, atol=1e-8, rtol=0)
    np.testing.assert_allclose(forbackiter, a, atol=1e-8, rtol=0)


def test_wrongdirection2():
    """
    testing if it gets the direction right and returns errors to user's wrong directions
    """
    # testing if it gets the direction right and returns errors to user's wrong directions
    vec: list[complex] =[0, 0, 0, 0]
    notdirections = ["hello", 2, [fft_direction.BACKWARD, fft_direction.FORWARD]]

    # testing invalid directions
    for direction in notdirections:
        try:
            fft_1d_recursive_radix2(vec, direction)
            assert False
        except ValueError:
            pass  # expected
        try:
            fft_1d_radix2(vec, direction)
            assert False
        except ValueError:
            pass

    # testing with valid directions
    try:
        fft_1d_recursive_radix2(vec, fft_direction.FORWARD)
        fft_1d_recursive_radix2(vec, fft_direction.BACKWARD)
        fft_1d_radix2(vec, fft_direction.FORWARD)
        fft_1d_radix2(vec, fft_direction.BACKWARD)
    except ValueError:
        assert False


def test_lotofnumbers2():
    """
    testing if my code can handle with very long lists
    """
    a: list[complex] = [
        (34.2 + 78.5j), (-12.4 + 9.8j), (56.1 - 23.3j), (-99.9 + 45.6j), (87.2 - 34.1j),
        (23.5 + 67.4j), (-45.3 + 12.2j), (19.8 - 76.4j), (90.1 + 11.3j), (-3.4 - 56.8j),
        (12.6 + 34.9j), (-67.2 - 89.3j), (45.1 + 78.2j), (23.4 - 56.7j), (-90.4 + 34.6j),
        (78.9 - 12.5j), (-34.7 + 23.8j), (11.1 - 99.2j), (65.3 + 45.2j), (-76.9 - 23.7j),
        (90.5 + 12.8j), (-88.6 + 76.1j), (34.5 - 89.7j), (12.2 + 23.6j), (-55.4 - 45.1j),
        (100.1 + 9.9j), (-23.6 - 78.3j), (56.7 + 34.8j), (45.9 - 67.2j), (-12.3 + 78.6j),
        (76.4 - 11.1j), (-34.8 + 99.9j), (19.7 - 56.3j), (90.8 + 45.6j), (-77.2 - 12.4j),
        (12.5 + 23.8j), (-56.3 - 78.5j), (34.1 + 99.6j), (89.7 - 45.8j), (-23.5 + 11.3j),
        (100.2 - 67.1j), (-56.8 + 78.9j), (12.4 - 45.6j), (78.5 + 34.7j), (-99.1 - 23.9j),
        (23.6 + 45.2j), (-67.4 - 12.8j), (12.1 + 78.3j), (56.2 - 34.9j), (-45.7 + 99.4j),
        (90.3 - 23.1j), (-78.6 + 12.5j), (34.2 - 67.8j), (12.3 + 100.5j), (-56.5 - 89.6j),
        (45.1 + 78.4j), (-67.2 - 12.9j), (100.6 + 23.7j), (-34.3 + 45.8j), (12.4 - 78.9j),
        (90.1 + 67.5j), (-99.2 - 45.3j), (56.4 + 23.8j), (78.6 - 12.7j), (-34.5 + 90.9j),
        (11.2 - 67.3j), (-45.8 + 100.8j), (12.9 - 23.4j), (67.5 + 78.2j), (-56.6 - 12.2j),
        (34.7 + 45.9j), (-78.9 - 99.5j), (100.9 + 12.1j), (-23.8 + 67.6j), (45.2 - 56.9j),
        (90.7 + 78.1j), (-34.6 - 23.3j), (12.5 + 100.7j), (-56.7 - 45.4j), (78.3 + 12.8j),
        (-99.8 - 67.7j), (34.9 + 23.5j), (-12.6 + 90.6j), (56.8 - 100.3j), (-78.4 + 45.7j),
        (90.2 - 12.9j), (-23.9 + 56.5j), (67.8 - 34.4j), (-45.3 + 78.7j), (12.7 - 100.2j),
        (34.8 + 67.9j), (-99.7 - 23.6j), (56.9 + 12.3j), (78.1 - 90.5j), (-100.4 + 45.6j),
        (-12.8 + 54.6j), (91.4 - 38.7j), (66.5 + 77.3j), (-45.6 + 21.1j),
        (14.9 - 92.8j), (-87.3 + 18.6j), (53.7 - 49.5j), (-26.4 - 35.2j),
        (77.1 + 11.2j), (-68.9 - 80.6j), (39.2 + 45.1j), (100.5 - 67.4j),
        (-55.1 + 34.3j), (88.6 - 99.9j), (-12.2 + 78.5j), (67.4 - 13.6j),
        (23.1 + 45.8j), (-44.7 - 56.2j), (89.3 + 24.7j), (-98.6 + 90.1j),
        (35.4 - 23.8j), (-100.0 + 78.3j), (12.2 - 33.7j), (56.7 + 67.9j),
        (-66.6 - 11.4j), (45.3 + 88.5j), (91.6 - 44.1j), (-32.7 + 56.8j),
        (100.3 - 99.2j), (18.8 + 12.3j), (-78.2 + 90.4j), (-92.13+20.13j), (92.13-20.13j)]
    n: int = len(a)
    # Forward functions
    b: list[complex] = fft_1d_recursive_radix2(a, fft_direction.FORWARD)
    c: list[complex] = fft_1d_radix2(a, fft_direction.FORWARD)
    # Backward functions
    d: list[complex] = fft_1d_recursive_radix2(a, fft_direction.BACKWARD)
    e: list[complex] = fft_1d_radix2(a, fft_direction.BACKWARD)
    # Desired output
    f: list[complex] = np.fft.fft(a) / n  # we do the little change to match our given formula
    i: list[complex] = np.fft.ifft(a) * n  # we do the little change to match our given formula
    np.testing.assert_allclose(b, c, atol=1e-4, rtol=0)
    np.testing.assert_allclose(b, i, atol=1e-4, rtol=0)
    np.testing.assert_allclose(d, e, atol=1e-4, rtol=0)
    np.testing.assert_allclose(d, f, atol=1e-4, rtol=0)


def test_overall2():
    """
    testing the functions in project 2 with a normal case
    """
    a: list[complex] = [4.32 + 5j, -1 - 3j, 1.47j, -6.32 + 2j]
    n: int = len(a)

    # our functions
    recfor = fft_1d_recursive_radix2(a, fft_direction.FORWARD)
    recback = fft_1d_recursive_radix2(a, fft_direction.BACKWARD)
    iterfor = fft_1d_radix2(a, fft_direction.FORWARD)
    iterback = fft_1d_radix2(a, fft_direction.BACKWARD)

    # what we desire to get, we change the numpy to be fitting the formula for fft we are using
    wantedback = np.fft.fft(a) / n
    wantedfor = np.fft.ifft(a) * n

    np.testing.assert_allclose(recfor, wantedfor, atol=1e-8, rtol=0)
    np.testing.assert_allclose(iterfor, wantedfor, atol=1e-8, rtol=0)
    np.testing.assert_allclose(recback, wantedback, atol=1e-8, rtol=0)
    np.testing.assert_allclose(iterback, wantedback, atol=1e-8, rtol=0)


def test_notvec2():
    """
    checking if the input checker for the vec is correct in each function from
    project 2
    """
    notarray: int = 55214142123146
    stringarray: list[object] = [5j, 3, "hello"]
    tuplenotarray: tuple[complex, complex, complex] = (5j, 3j, 10j)
    notvecs = [notarray, stringarray, tuplenotarray]  # for easier use later with iteration
    for notvec in notvecs:
        try:
            fft_1d_recursive_radix2(notvec, fft_direction.FORWARD)
            assert False  # If no error happens, the test should fail
        except TypeError:
            pass  # If an error happens, that's what we want

    for notvec in notvecs:
        try:
            fft_1d_radix2(notvec, fft_direction.FORWARD)
            assert False
        except TypeError:
            pass
    assert True


def test_oddvec2():
    """
    tests if the functions in project 2 behave as wanted when the vec is of odd
    number of elements
    """
    oddvecs: list[list[complex]] = [[0, 0, 0], [1, 2, 3, 4, 5], [1j, 4j, 0j, 4.32, -3, 0, 0, 0, 0]]
    for vec in oddvecs:
        try:
            fft_1d_recursive_radix2(vec, fft_direction.FORWARD)
            assert False # if no error happens, the test should fail
        except ValueError:
            pass
    for vec in oddvecs:
        try:
            fft_1d_recursive_radix2(vec, fft_direction.FORWARD)
            assert False
        except ValueError:
            pass
    assert True


def test_bigvalues2():
    """
    testing our functions in project 2 using big values
    """
    vec: list[complex] = [(2**8)*1j, -(2**9)*1j]
    # our fft
    recfor = fft_1d_recursive_radix2(vec, fft_direction.FORWARD)
    recback = fft_1d_recursive_radix2(vec, fft_direction.BACKWARD)
    iterfor = fft_1d_radix2(vec, fft_direction.FORWARD)
    iterback = fft_1d_radix2(vec, fft_direction.BACKWARD)

    # numpy functions after changing them a bit to fit the formula we use
    wantedback = np.fft.fft(vec) / 2  # len(vec)=2  #
    wantedfor = np.fft.ifft(vec) * 2  # len(vec)=2

    np.testing.assert_allclose(recfor, wantedfor, atol=1e-8, rtol=0)
    np.testing.assert_allclose(recback, wantedback, atol=1e-8, rtol=0)
    np.testing.assert_allclose(iterfor, wantedfor, atol=1e-8, rtol=0)
    np.testing.assert_allclose(iterback, wantedback, atol=1e-8, rtol=0)


def test_inversability1():
    """
    checking if it kind of returns to itself after going both ways of the function
    in project 1
    """
    a: list[complex] = [1j, 2j, 3j, 4j]
    afor: list[complex] = dft_1d(a, fft_direction.FORWARD)
    forback: list[complex] = dft_1d(afor, fft_direction.BACKWARD)
    aformatrix: list[complex] = dft_1d_matrix(a, fft_direction.FORWARD)
    forbackm: list[complex] = dft_1d_matrix(aformatrix, fft_direction.BACKWARD)

    if (abs(forback[0] - a[0]) < 1e-8) & (abs(forback[1] - a[1]) < 1e-8) &\
            (abs(forback[2] - a[2]) < 1e-8) & (abs(forback[3] - a[3]) < 1e-8) &\
            (abs(forbackm[0] - a[0]) < 1e-8) & (abs(forbackm[1] - a[1]) < 1e-8) &\
            (abs(forbackm[2] - a[2]) < 1e-8) & (abs(forbackm[3] - a[3]) < 1e-8):
        # we are checking with consideration to little differences because of the system
        assert True
    else:
        assert False


def test_wrongdirection1():
    """
    testing project 1 functions if it gets the direction right and returns
    errors to user's wrong directions
    """
    hi: str = "hello"
    if (dft_1d([0, 0, 0, 0], hi) != 0) | (dft_1d([0, 0, 0, 0], 2) != 0) | (
        dft_1d([0, 0, 0, 0], fft_direction.FORWARD) == 0) |\
            (dft_1d([0, 0, 0, 0], fft_direction.BACKWARD) == 0)\
            | (dft_1d_matrix([0, 0, 0, 0], hi) != 0) |\
            (dft_1d_matrix([0, 0, 0, 0], 2) != 0) | (
                dft_1d_matrix([0, 0, 0, 0], fft_direction.FORWARD) == 0) |\
            (dft_1d_matrix([0, 0, 0, 0], fft_direction.BACKWARD) == 0):
        assert False
    else:
        assert True


def test_single1():
    """
    testing project 1 functions in extreme situations, such as the array has only one element
    """
    a: list[complex] = [1j]
    # In such cases, the difference between numpy functions and ours is switching
    # between the inverse and the forward
    b: list[complex] = [1 + 0j]
    c: list[complex] = [1 + 1j]
    if not (abs(dft_1d(a, fft_direction.FORWARD) - np.fft.ifft(a)) < 1e-8) & (
            abs(dft_1d(b, fft_direction.FORWARD) - np.fft.ifft(b)) < 1e-8):
        assert False
    if abs(dft_1d(c, fft_direction.FORWARD) - np.fft.ifft(c)) > 1e-8:
        assert False
    if not (abs(dft_1d_matrix(a, fft_direction.FORWARD) - np.fft.ifft(a)) < 1e-8) & (
            abs(dft_1d_matrix(b, fft_direction.FORWARD) - np.fft.ifft(b)) < 1e-8):
        assert False
    if abs(dft_1d_matrix(c, fft_direction.FORWARD) - np.fft.ifft(c)) > 1e-8:
        assert False
    assert True


def test_lotofnumbers1():
    """
    testing if my project1 functions can handle with very long lists

    """
    a: list[complex] = [
            (34.2 + 78.5j), (-12.4 + 9.8j), (56.1 - 23.3j), (-99.9 + 45.6j), (87.2 - 34.1j),
            (23.5 + 67.4j), (-45.3 + 12.2j), (19.8 - 76.4j), (90.1 + 11.3j), (-3.4 - 56.8j),
            (12.6 + 34.9j), (-67.2 - 89.3j), (45.1 + 78.2j), (23.4 - 56.7j), (-90.4 + 34.6j),
            (78.9 - 12.5j), (-34.7 + 23.8j), (11.1 - 99.2j), (65.3 + 45.2j), (-76.9 - 23.7j),
            (90.5 + 12.8j), (-88.6 + 76.1j), (34.5 - 89.7j), (12.2 + 23.6j), (-55.4 - 45.1j),
            (100.1 + 9.9j), (-23.6 - 78.3j), (56.7 + 34.8j), (45.9 - 67.2j), (-12.3 + 78.6j),
            (76.4 - 11.1j), (-34.8 + 99.9j), (19.7 - 56.3j), (90.8 + 45.6j), (-77.2 - 12.4j),
            (12.5 + 23.8j), (-56.3 - 78.5j), (34.1 + 99.6j), (89.7 - 45.8j), (-23.5 + 11.3j),
            (100.2 - 67.1j), (-56.8 + 78.9j), (12.4 - 45.6j), (78.5 + 34.7j), (-99.1 - 23.9j),
            (23.6 + 45.2j), (-67.4 - 12.8j), (12.1 + 78.3j), (56.2 - 34.9j), (-45.7 + 99.4j),
            (90.3 - 23.1j), (-78.6 + 12.5j), (34.2 - 67.8j), (12.3 + 100.5j), (-56.5 - 89.6j),
            (45.1 + 78.4j), (-67.2 - 12.9j), (100.6 + 23.7j), (-34.3 + 45.8j), (12.4 - 78.9j),
            (90.1 + 67.5j), (-99.2 - 45.3j), (56.4 + 23.8j), (78.6 - 12.7j), (-34.5 + 90.9j),
            (11.2 - 67.3j), (-45.8 + 100.8j), (12.9 - 23.4j), (67.5 + 78.2j), (-56.6 - 12.2j),
            (34.7 + 45.9j), (-78.9 - 99.5j), (100.9 + 12.1j), (-23.8 + 67.6j), (45.2 - 56.9j),
            (90.7 + 78.1j), (-34.6 - 23.3j), (12.5 + 100.7j), (-56.7 - 45.4j), (78.3 + 12.8j),
            (-99.8 - 67.7j), (34.9 + 23.5j), (-12.6 + 90.6j), (56.8 - 100.3j), (-78.4 + 45.7j),
            (90.2 - 12.9j), (-23.9 + 56.5j), (67.8 - 34.4j), (-45.3 + 78.7j), (12.7 - 100.2j),
            (34.8 + 67.9j), (-99.7 - 23.6j), (56.9 + 12.3j), (78.1 - 90.5j), (-100.4 + 45.6j)
        ]
    n: int = len(a)
    b: list[complex] = dft_1d(a, fft_direction.FORWARD)
    c: list[complex] = dft_1d_matrix(a, fft_direction.FORWARD)
    d: list[complex] = dft_1d(a, fft_direction.BACKWARD)
    e: list[complex] = dft_1d_matrix(a, fft_direction.BACKWARD)
    f: list[complex] = np.fft.fft(a) / n  # we do the little change to match our given formula
    i: list[complex] = np.fft.ifft(a) * n  # we do the little change to match our given formula
    if (all(abs(x - y) <= 10e-4 for x, y in zip(b, c))) &\
            (all(abs(x - y) <= 10e-4 for x, y in zip(b, i))) & (
            all(abs(x - y) <= 10e-4 for x, y in zip(d, e))) &\
            (all(abs(x - y) <= 10e-4 for x, y in zip(d, f))):
        assert True
        # we are checking our functions against the known numpy corresponding functions
    else:
        assert False


def test_overall1():
    """
    testing project1 functions with normal cases
    """
    a: list[complex] = [4.32 + 5j, -1 - 3j, 1.47j, -6.32 + 2j]
    n: int = len(a)
    ourinvresult: list[complex] = dft_1d(a, fft_direction.BACKWARD)
    ourresult: list[complex] = dft_1d(a, fft_direction.FORWARD)
    fixednumpyresult: list[complex] = np.fft.fft(a) / n
    fixedinvnumpyresult: list[complex] = np.fft.ifft(a) * n
    if (max(list(map(abs, ourresult - fixedinvnumpyresult))) < 1e-8) & (
            max(list(map(abs, ourinvresult - fixednumpyresult))) < 1e-8):
        # we were checking if our function does it right like the numpy function
        # after we do the little changebetween their and our formulas with the /n and *n
        assert True
    else:
        assert False


def test_notvec1():
    """
    checking if the input checker for vec is correct in project1 functions
    :return:
    """
    notarray: int = 55214142123146
    stringarray: list[object] = [5j, 3, "hello"]
    tuplenotarray: tuple[complex, complex, complex] = (5j, 3j, 10j)
    if (not dft_1d(notarray, fft_direction.FORWARD) is False) |\
            (not dft_1d(stringarray, fft_direction.FORWARD) is False) |\
            (not dft_1d(tuplenotarray, fft_direction.FORWARD) is False):
        assert False  # checking if it returns the errors.
    if (not dft_1d_matrix(notarray, fft_direction.FORWARD) is False) | \
            (not dft_1d_matrix(stringarray, fft_direction.FORWARD) is False) | \
            (not dft_1d_matrix(tuplenotarray, fft_direction.FORWARD) is False):
        assert False
    assert True


def test_bigvalues1():
    """
    testing our project1 functions when using large values
    """
    vec: list[complex] = [(2**8)*1j, -(2**9)*1j]
    normalnewvecfor: list[complex] = dft_1d(vec, fft_direction.FORWARD)
    normalnewvecback: list[complex] = dft_1d(vec, fft_direction.BACKWARD)
    matrixnewvecfor: list[complex] = dft_1d_matrix(vec, fft_direction.FORWARD)
    matrixnewvecback: list[complex] = dft_1d_matrix(vec, fft_direction.BACKWARD)
    fixednumpyresult: list[complex] = np.fft.fft(vec) / 2
    fixedinvnumpyresult: list[complex] = np.fft.ifft(vec) * 2
    if (max(list(map(abs, normalnewvecfor - fixedinvnumpyresult))) < 1e-8) & (
            max(list(map(abs, normalnewvecback - fixednumpyresult))) < 1e-8) & \
            (max(list(map(abs, matrixnewvecfor - fixedinvnumpyresult))) < 1e-8) &\
            (max(list(map(abs, matrixnewvecback - fixednumpyresult))) < 1e-8):

        assert True
    else:
        assert False
