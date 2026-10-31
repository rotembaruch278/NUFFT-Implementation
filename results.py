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
from nufft import nufft_1d_I, nufft_1d_II, nufft_1d_III, measure_error_2, measure_error_inf
from nufft import naive_nufft1d_I, naive_nufft1d_II, naive_nufft1d_III, fft_1d, fft_direction


def getinfoforN1a(N: int) -> list:
    """
    We use this function to get all the information for table 1A
    given a number N for the arrays sizes.
    """
    # First we store all the time info, on the way we get all the
    # fft's nufft_1d_I and the naive done.
    timeAlgoinitstart: np.float64 = time.perf_counter()
    alpha: np.ndarray[np.complex64] = np.random.rand(N+1) + 1j * np.random.rand(N+1)
    w = np.random.uniform(-N / 2, N / 2, N + 1)  # frequencies
    timeAlgoinitend: np.float64 = time.perf_counter()
    timeAlgoinit: np.float64 = timeAlgoinitend - timeAlgoinitstart
    timeAlgoEvalstart: np.float64 = time.perf_counter()
    result = nufft_1d_I(alpha, w, 'single')
    timeAlgoEvalend: np.float64 = time.perf_counter()
    timeAlgoEval: np.float64 = timeAlgoEvalend - timeAlgoEvalstart
    timeNaivestart: np.float64 = time.perf_counter()
    resultnaivedouble: np.ndarray[np.complex128] = naive_nufft1d_I(alpha, w, 'double')
    timeNaiveend: np.float64 = time.perf_counter()
    timeNaive: np.float64 = timeNaiveend - timeNaivestart
    timefftstart: np.float64 = time.perf_counter()
    trashresult: np.complex64 = fft_1d(alpha,direction=fft_direction.FORWARD)
    timefftend: np.float64 = time.perf_counter()
    timefft: np.float64 = timefftend - timefftstart
    # Now we check the errors
    e2Algo: np.float64 = measure_error_2(result, resultnaivedouble)
    einfAlgo: np.float64 = measure_error_inf(result, resultnaivedouble, alpha)
    resultnaivesingle: np.ndarray[np.complex64] = naive_nufft1d_I(alpha, w, 'single')
    e2Direct: np.float64 = measure_error_2(resultnaivesingle, resultnaivedouble)
    einfDirect: np.float64 = measure_error_inf(resultnaivesingle, resultnaivedouble, alpha)
    return [N, f"{einfAlgo:.3E}", f"{e2Algo:.3E}", f"{einfDirect:.3E}",
            f"{e2Direct:.3E}", f"{timeAlgoinit:.3E}", f"{timeAlgoEval:.3E}",
            f"{timeNaive:.2E}", f"{timefft:.4E}"]


def Table1a():
    """
    Creating the Table 1(a) from the paper,
    we use function getinfoforN1a couple of times
    to get each row efficiently

    """
    columns: list = [
        "N", "Alg E∞", "Alg E2", "Direct E∞", "Direct E2",
        "Timing Init", "Timing Eval", "Timing Direct", "Timing FFT"
    ]
    # In order to make it efficient, every time we will run all the checks on a different N:

    firstrow: list = getinfoforN1a(64)
    secondrow: list = getinfoforN1a(128)
    thirdrow: list = getinfoforN1a(256)
    fourthrow: list = getinfoforN1a(512)
    fifthrow: list = getinfoforN1a(1024)
    sixthrow: list = getinfoforN1a(2048)
    seventhrow: list = getinfoforN1a(4096)
    rows = [firstrow, secondrow, thirdrow, fourthrow, fifthrow, sixthrow, seventhrow]
    # Create figure
    fig, ax = mp.subplots(figsize=(9, 3))
    ax.axis('off')

    # Make table
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # scale for readability

    mp.title("Example 1: Single precision computations", fontsize=12, pad=10)
    mp.show()


def getinfoforN1b(N: int) -> list:
    """
    We use this function to get all the information for table 1b
    given a number N for the arrays sizes.
    """
    # We start with measuring all the time factors and on the way save results

    timeAlgoinitstart: np.float64 = time.perf_counter()
    alpha: np.ndarray[np.complex128] = np.random.rand(N + 1) + 1j * np.random.rand(N + 1)
    w = np.random.uniform(-N / 2, N / 2, N + 1)  # frequencies
    timeAlgoinitend: np.float64 = time.perf_counter()
    timeAlgoinit: np.float64 = timeAlgoinitend - timeAlgoinitstart
    timeAlgoEvalstart: np.float64 = time.perf_counter()
    result = nufft_1d_I(alpha, w, 'double')
    timeAlgoEvalend: np.float64 = time.perf_counter()
    timeAlgoEval: np.float64 = timeAlgoEvalend - timeAlgoEvalstart
    timeNaivestart: np.float64 = time.perf_counter()
    resultnaivedouble: np.ndarray[np.complex128] = naive_nufft1d_I(alpha, w, 'double')
    timeNaiveend: np.float64 = time.perf_counter()
    timeNaive: np.float64 = timeNaiveend - timeNaivestart
    timefftstart: np.float64 = time.perf_counter()
    trashresult: np.complex64 = fft_1d(alpha, direction=fft_direction.FORWARD)
    timefftend: np.float64 = time.perf_counter()
    timefft: np.float64 = timefftend - timefftstart
    # Now we check the errors
    e2Algo: np.float64 = measure_error_2(result, resultnaivedouble)
    einfAlgo: np.float64 = measure_error_inf(result, resultnaivedouble, alpha)
    return [N, f"{einfAlgo:.3E}", f"{e2Algo:.3E}",
             f"{timeAlgoinit:.3E}", f"{timeAlgoEval:.3E}",
            f"{timeNaive:.2E}", f"{timefft:.4E}"]

def Table1b():
    """
    Creating the Table 1(b) from the paper,
    we use function getinfoforN1b couple of times
    to get each row efficiently

    """
    columns: list = [
        "N", "Alg E∞", "Alg E2",
        "Timing Init", "Timing Eval", "Timing Direct", "Timing FFT"
    ]
    # In order to make it efficient, every time we will run all the checks on a different N:

    firstrow: list = getinfoforN1b(64)
    secondrow: list = getinfoforN1b(128)
    thirdrow: list = getinfoforN1b(256)
    fourthrow: list = getinfoforN1b(512)
    fifthrow: list = getinfoforN1b(1024)
    sixthrow: list = getinfoforN1b(2048)
    seventhrow: list = getinfoforN1b(4096)
    rows = [firstrow, secondrow, thirdrow, fourthrow, fifthrow, sixthrow, seventhrow]
    # Create figure
    fig, ax = mp.subplots(figsize=(7, 3))
    ax.axis('off')

    # Make table
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # scale for readability

    mp.title("Example 1: Double precision computations", fontsize=12, pad=10)
    mp.show()


def getinfoforN2a(N: int) -> list:
    """
    We use this function to get all the information for table 2A
    given a number N for the arrays sizes.
    """
    # First we store all the time info, on the way we get all the
    # fft's nufft_1d_II and the naive done.
    timeAlgoinitstart: np.float64 = time.perf_counter()
    beta: np.ndarray[np.complex64] = np.random.rand(N+1) + 1j * np.random.rand(N+1)
    x = np.random.uniform(-np.pi, np.pi, N + 1)
    timeAlgoinitend: np.float64 = time.perf_counter()
    timeAlgoinit: np.float64 = timeAlgoinitend - timeAlgoinitstart
    timeAlgoEvalstart: np.float64 = time.perf_counter()
    result = nufft_1d_II(beta, x, 'single')
    timeAlgoEvalend: np.float64 = time.perf_counter()
    timeAlgoEval: np.float64 = timeAlgoEvalend - timeAlgoEvalstart
    timeNaivestart: np.float64 = time.perf_counter()
    resultnaivedouble: np.ndarray[np.complex128] = naive_nufft1d_II(beta, x, 'double')
    timeNaiveend: np.float64 = time.perf_counter()
    timeNaive: np.float64 = timeNaiveend - timeNaivestart
    timefftstart: np.float64 = time.perf_counter()
    trashresult: np.complex64 = fft_1d(beta, direction=fft_direction.FORWARD)
    timefftend: np.float64 = time.perf_counter()
    timefft: np.float64 = timefftend - timefftstart
    # Now measuring the errors
    e2Algo: np.float64 = measure_error_2(result, resultnaivedouble)
    einfAlgo: np.float64 = measure_error_inf(result, resultnaivedouble, beta)
    resultnaivesingle: np.ndarray[np.complex64] = naive_nufft1d_II(beta, x, 'single')
    e2Direct: np.float64 = measure_error_2(resultnaivesingle, resultnaivedouble)
    einfDirect: np.float64 = measure_error_inf(resultnaivesingle, resultnaivedouble, beta)
    return [N, f"{einfAlgo:.3E}", f"{e2Algo:.3E}", f"{einfDirect:.3E}",
            f"{e2Direct:.3E}", f"{timeAlgoinit:.3E}", f"{timeAlgoEval:.3E}",
            f"{timeNaive:.2E}", f"{timefft:.4E}"]



def Table2a():
    """
    Creating the Table 2(a) from the paper,
    we use function getinfoforN2a couple of times
    to get each row efficiently

    """
    columns: list = [
        "N", "Alg E∞", "Alg E2", "Direct E∞", "Direct E2",
        "Timing Init", "Timing Eval", "Timing Direct", "Timing FFT"
    ]
    # In order to make it efficient, every time we will run all the checks on a different N:

    firstrow: list = getinfoforN2a(64)
    secondrow: list = getinfoforN2a(128)
    thirdrow: list = getinfoforN2a(256)
    fourthrow: list = getinfoforN2a(512)
    fifthrow: list = getinfoforN2a(1024)
    sixthrow: list = getinfoforN2a(2048)
    seventhrow: list = getinfoforN2a(4096)
    rows = [firstrow, secondrow, thirdrow, fourthrow, fifthrow, sixthrow, seventhrow]
    # Create figure
    fig, ax = mp.subplots(figsize=(9, 3))
    ax.axis('off')

    # Make table
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # scale for readability

    mp.title("Example 2: Single precision computations", fontsize=12, pad=10)
    mp.show()


def getinfoforN2b(N: int) -> list:
    """
    We use this function to get all the information for table 2b
    given a number N for the arrays sizes.
    """
    # First we measure all the time thingies and on the way
    # the transformations values
    timeAlgoinitstart: np.float64 = time.perf_counter()
    beta: np.ndarray[np.complex128] = np.random.rand(N + 1) + 1j * np.random.rand(N + 1)
    x = np.random.uniform(-np.pi, np.pi, N + 1)
    timeAlgoinitend: np.float64 = time.perf_counter()
    timeAlgoinit: np.float64 = timeAlgoinitend - timeAlgoinitstart
    timeAlgoEvalstart: np.float64 = time.perf_counter()
    result = nufft_1d_II(beta, x, 'double')
    timeAlgoEvalend: np.float64 = time.perf_counter()
    timeAlgoEval: np.float64 = timeAlgoEvalend - timeAlgoEvalstart
    timeNaivestart: np.float64 = time.perf_counter()
    resultnaivedouble: np.ndarray[np.complex128] = naive_nufft1d_II(beta, x, 'double')
    timeNaiveend: np.float64 = time.perf_counter()
    timeNaive: np.float64 = timeNaiveend - timeNaivestart
    timefftstart: np.float64 = time.perf_counter()
    trashresult: np.complex64 = fft_1d(beta, direction=fft_direction.FORWARD)
    timefftend: np.float64 = time.perf_counter()
    timefft: np.float64 = timefftend - timefftstart
    # Now measuring the errors
    e2Algo: np.float64 = measure_error_2(result, resultnaivedouble)
    einfAlgo: np.float64 = measure_error_inf(result, resultnaivedouble, beta)
    return [N, f"{einfAlgo:.3E}", f"{e2Algo:.3E}",
             f"{timeAlgoinit:.3E}", f"{timeAlgoEval:.3E}",
            f"{timeNaive:.2E}", f"{timefft:.4E}"]


def Table2b():
    """
    Creating the Table 2(b) from the paper,
    we use function getinfoforN2b couple of times
    to get each row efficiently

    """
    columns: list = [
        "N", "Alg E∞", "Alg E2",
        "Timing Init", "Timing Eval", "Timing Direct", "Timing FFT"
    ]
    # In order to make it efficient, every time we will run all the checks on a different N:

    firstrow: list = getinfoforN2b(64)
    secondrow: list = getinfoforN2b(128)
    thirdrow: list = getinfoforN2b(256)
    fourthrow: list = getinfoforN2b(512)
    fifthrow: list = getinfoforN2b(1024)
    sixthrow: list = getinfoforN2b(2048)
    seventhrow: list = getinfoforN2b(4096)
    rows = [firstrow, secondrow, thirdrow, fourthrow, fifthrow, sixthrow, seventhrow]
    # Create figure
    fig, ax = mp.subplots(figsize=(7, 3))
    ax.axis('off')

    # Make table
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # scale for readability

    mp.title("Example 2: Double precision computations", fontsize=12, pad=10)
    mp.show()


def getinfoforN3a(N: int) -> list:
    """
    We use this function to get all the information for table 2A
    given a number N for the arrays sizes.
    """
    # First we store all the time info, on the way we get all the
    # fft's nufft_1d_III and the naive done.
    timeAlgoinitstart: np.float64 = time.perf_counter()
    gamma: np.ndarray[np.complex64] = np.random.rand(N+1) + 1j * np.random.rand(N+1)
    x = np.random.uniform(-np.pi, np.pi, N + 1)
    w = np.random.uniform(-N / 2, N / 2, N + 1)  # frequencies
    timeAlgoinitend: np.float64 = time.perf_counter()
    timeAlgoinit: np.float64 = timeAlgoinitend - timeAlgoinitstart
    timeAlgoEvalstart: np.float64 = time.perf_counter()
    result = nufft_1d_III(gamma, w, x, 'single')
    timeAlgoEvalend: np.float64 = time.perf_counter()
    timeAlgoEval: np.float64 = timeAlgoEvalend - timeAlgoEvalstart
    timeNaivestart: np.float64 = time.perf_counter()
    resultnaivedouble: np.ndarray[np.complex128] = naive_nufft1d_III(gamma, w, x, 'double')
    timeNaiveend: np.float64 = time.perf_counter()
    timeNaive: np.float64 = timeNaiveend - timeNaivestart
    timefftstart: np.float64 = time.perf_counter()
    trashresult: np.complex64 = fft_1d(gamma, direction=fft_direction.FORWARD)
    timefftend: np.float64 = time.perf_counter()
    timefft: np.float64 = timefftend - timefftstart
    # Now measuring the errors
    e2Algo: np.float64 = measure_error_2(result, resultnaivedouble)
    einfAlgo: np.float64 = measure_error_inf(result, resultnaivedouble, gamma)
    resultnaivesingle: np.ndarray[np.complex64] = naive_nufft1d_III(gamma, w, x, 'single')
    e2Direct: np.float64 = measure_error_2(resultnaivesingle, resultnaivedouble)
    einfDirect: np.float64 = measure_error_inf(resultnaivesingle, resultnaivedouble, gamma)
    return [N, f"{einfAlgo:.3E}", f"{e2Algo:.3E}", f"{einfDirect:.3E}",
            f"{e2Direct:.3E}", f"{timeAlgoinit:.3E}", f"{timeAlgoEval:.3E}",
            f"{timeNaive:.2E}", f"{timefft:.4E}"]


def Table3a():
    """
    Creating the Table 3(a) from the paper,
    we use function getinfoforN3a couple of times
    to get each row efficiently

    """
    columns: list = [
        "N", "Alg E∞", "Alg E2", "Direct E∞", "Direct E2",
        "Timing Init", "Timing Eval", "Timing Direct", "Timing FFT"
    ]
    # In order to make it efficient, every time we will run all the checks on a different N:

    firstrow: list = getinfoforN3a(64)
    secondrow: list = getinfoforN3a(128)
    thirdrow: list = getinfoforN3a(256)
    fourthrow: list = getinfoforN3a(512)
    fifthrow: list = getinfoforN3a(1024)
    sixthrow: list = getinfoforN3a(2048)
    seventhrow: list = getinfoforN3a(4096)
    rows = [firstrow, secondrow, thirdrow, fourthrow, fifthrow, sixthrow, seventhrow]
    # Create figure
    fig, ax = mp.subplots(figsize=(9, 3))
    ax.axis('off')

    # Make table
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # scale for readability

    mp.title("Example 3: Single precision computations", fontsize=12, pad=10)
    mp.show()


def getinfoforN3b(N):
    """
    We use this function to get all the information for table 3b
    given a number N for the arrays sizes.
    """
    # We measure the run times of the algorithm and on the way
    # save the values
    timeAlgoinitstart: np.float64 = time.perf_counter()
    gamma: np.ndarray[np.complex128] = np.random.rand(N + 1) + 1j * np.random.rand(N + 1)
    x = np.random.uniform(-np.pi, np.pi, N + 1)
    w = np.random.uniform(-N / 2, N / 2, N + 1)  # frequencies

    timeAlgoinitend: np.float64 = time.perf_counter()
    timeAlgoinit: np.float64 = timeAlgoinitend - timeAlgoinitstart
    timeAlgoEvalstart: np.float64 = time.perf_counter()
    result = nufft_1d_III(gamma, w, x, 'double')
    timeAlgoEvalend: np.float64 = time.perf_counter()
    timeAlgoEval: np.float64 = timeAlgoEvalend - timeAlgoEvalstart
    timeNaivestart: np.float64 = time.perf_counter()
    resultnaivedouble: np.ndarray[np.complex128] = naive_nufft1d_III(gamma, w, x, 'double')
    timeNaiveend: np.float64 = time.perf_counter()
    timeNaive: np.float64 = timeNaiveend - timeNaivestart
    timefftstart: np.float64 = time.perf_counter()
    trashresult: np.complex64 = fft_1d(gamma, direction=fft_direction.FORWARD)
    timefftend: np.float64 = time.perf_counter()
    timefft: np.float64 = timefftend - timefftstart
    # Now we measure errors
    e2Algo: np.float64 = measure_error_2(result, resultnaivedouble)
    einfAlgo: np.float64 = measure_error_inf(result, resultnaivedouble, gamma)
    return [N, f"{einfAlgo:.3E}", f"{e2Algo:.3E}",
             f"{timeAlgoinit:.3E}", f"{timeAlgoEval:.3E}",
            f"{timeNaive:.2E}", f"{timefft:.4E}"]


def Table3b():
    """
    Creating the Table 3(b) from the paper,
    we use function getinfoforN3b couple of times
    to get each row efficiently

    """
    columns: list = [
        "N", "Alg E∞", "Alg E2",
        "Timing Init", "Timing Eval", "Timing Direct", "Timing FFT"
    ]
    # In order to make it efficient, every time we will run all the checks on a different N:

    firstrow: list = getinfoforN3b(64)
    secondrow: list = getinfoforN3b(128)
    thirdrow: list = getinfoforN3b(256)
    fourthrow: list = getinfoforN3b(512)
    fifthrow: list = getinfoforN3b(1024)
    sixthrow: list = getinfoforN3b(2048)
    seventhrow: list = getinfoforN3b(4096)
    rows = [firstrow, secondrow, thirdrow, fourthrow, fifthrow, sixthrow, seventhrow]
    # Create figure
    fig, ax = mp.subplots(figsize=(7, 3))
    ax.axis('off')

    # Make table
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # scale for readability

    mp.title("Example 3: Double precision computations", fontsize=12, pad=10)
    mp.show()


Table1a()
Table1b()
Table2a()
Table2b()
Table3a()
Table3b()