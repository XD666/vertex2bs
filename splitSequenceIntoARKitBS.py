import numpy as np
from cvxopt import matrix, solvers
import vtk
import glob, os
import pandas as pd
oj = os.path.join

def vtkReadMesh(filename): 
    if filename.lower().endswith('.ply'): 
        reader = vtk.vtkPLYReader()
    elif filename.lower().endswith('.obj'): 
        reader = vtk.vtkOBJReader()
    else: 
        raise ValueError("ERROR: currenlyt only supporting .ply and .obj, unsupported file format: {}".format(filename))
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def writePly(vtkPolyData, filename):
    """Write mesh as ply file."""
    writer = vtk.vtkPLYWriter()
    writer.SetInputData(vtkPolyData)
    #writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
    writer.Write()

############################# test on single mesh ############################
# build BS info 
BSFolder = 'correctScale_mappedToTemplate/BS'
basisFile = 'correctScale_mappedToTemplate/Base.obj'
vtp_basis = vtkReadMesh(basisFile)
basis = np.array(vtp_basis.GetPoints().GetData())

fLs = glob.glob(oj(BSFolder, '*.obj'))
bsShapeLs = []
bsNameLs = []
for f in sorted(fLs): 
    _, fName = os.path.split(f)
    bsName = fName[:-4]
    bsNameLs.append(bsName)

    vtp = vtkReadMesh(f)
    vertices = np.array(vtp.GetPoints().GetData()) # 5023x3
    bsShapeLs.append(vertices.flatten())

bsNameArr = np.array(bsNameLs)
bsShapeArr = np.array(bsShapeLs) # 52x15069

#######################################################
def ridge(X, y, l2):
    """Ridge Regression model with intercept term.
    L2 penalty and intercept term included via design matrix augmentation.
    This augmentation allows for the OLS estimator to be used for fitting.
    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        l2 - L2 penalty tuning parameter (positive scalar) 
    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    m, n = np.shape(X)
    upper_half = np.hstack((np.ones((m, 1)), X))
    lower = np.zeros((n, n))
    np.fill_diagonal(lower, np.sqrt(l2))
    lower_half = np.hstack((np.zeros((n, 1)), lower))
    X = np.vstack((upper_half, lower_half))
    y = np.append(y, np.zeros(n))
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


def splitBSCoef_ridge(basis, bsShapeArr, targetNeutral, target): 
    """https://towardsdatascience.com/regularized-linear-regression-models-44572e79a1b5"""
    A = bsShapeArr.T - np.repeat(basis.flatten().reshape(-1, 1), len(bsShapeArr), 1)
    b = target.flatten() - targetNeutral.flatten()

    reg = 0.000001 # L2 regularization term
    x = ridge(A, b, reg) # 52+1 dimensional
    return x[:-1]

def splitBSCoef(basis, bsShapeArr, targetNeutral, target): 
    """returns 
    x           - solution coefs
    residual    - sums of squared residuals
    rank        - rank of matrix A
    s           - singular values of a"""
    A = bsShapeArr.T - np.repeat(basis.flatten().reshape(-1, 1), len(bsShapeArr), 1)
    b = target.flatten() - targetNeutral.flatten()
    x = np.linalg.lstsq(A, b)
    return x 


def splitBSCoef_qp(basis, bsShapeArr, targetNeutral, target): 
    """quadratic programming, which guarantees [0,1] output"""
    A = bsShapeArr.T - np.repeat(basis.flatten().reshape(-1, 1), len(bsShapeArr), 1)
    b = target.flatten() - targetNeutral.flatten()
    x = qp(A, b, 0, 1)
    return x 
    
def qp(A, b, min,    max, weight=None):
    """Solve constrained least square problem with quadratic programming.

    See: https://cvxopt.org/userguide/coneprog.html#quadratic-programming.
    """
    # A: (M,N), b: (M,) or (M,P)
    # return x: (N,) or (N,P)
    P = A.T.dot(A).astype(np.double)
    q = -A.T.dot(b).astype(np.double)
    
    if weight is not None:
        # https://online.stat.psu.edu/stat501/lesson/13/13.1
        # https://en.wikipedia.org/wiki/Generalized_least_squares
        # W = Omega^-1
        # 1/2 x^T P x + q^T x + 1/2 x^T diag(weight) x
        print('A^T A Norm (=max(svd(A))^2):', np.linalg.norm(P, ord=2))
        # print('eig(A^T A) max:', np.linalg.eig(P)[0].max())
        P += np.diag(weight) # weight should be in scale of eig(P=A^T A) or sigma(A)^2
        print('A^T A Norm with weight (=max(svd(A))^2):', np.linalg.norm(P, ord=2))

    I = np.eye(A.shape[1])
    G = np.vstack([-I, I])
    ones = np.ones(A.shape[1])
    h = np.stack([min * ones, max * ones]).ravel()

    solvers.options['show_progress'] = False
    if q.ndim == 1:
        res = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        return np.asarray(res['x']) # (N,)
    else:
        P = matrix(P.astype(np.double)) # cvxopt only accepts double, not int
        G = matrix(G.astype(np.double))
        h = matrix(h.astype(np.double))
        res = []
        for i in range(q.shape[1]):
            _res = solvers.qp(P, matrix(q[:,i]), G, h)
            res.append(np.asarray(_res['x']))
        return np.asarray(res).T.squeeze() # (N, P)

"""
test single mesh decompose
"""

targetNeutral_vtp = vtkReadMesh('FaceTalk_170904_03276_TA.ply')
targetNeutral = np.array(targetNeutral_vtp.GetPoints().GetData())

# target_vtp = vtkReadMesh('FaceTalk_170904_03276_TA.ply')
target_vtp = vtkReadMesh('sentence04/sentence04.000018.ply')
target = np.array(target_vtp.GetPoints().GetData())

x = splitBSCoef_qp(basis, bsShapeArr, targetNeutral, target)
x[x<1e-4] = 0
print(x)

# x, residual, rank, singular_value = splitBSCoef(basis, bsShapeArr, targetNeutral, target)
# print(x)
# x = x.clip(0, 1)
# print('after clipping to [0, 1]:')
# print(x)

delta = ((bsShapeArr - basis.flatten()) * x.reshape(-1, 1)).sum(axis=0).reshape(-1, 3) # 5023x3
shape = targetNeutral + delta


def changeCoord(vtp, coord): 
    """ change coord of all points 

    A new copy of vtp is created and returned. input vtp is not modified. """
    vtpNew = vtk.vtkPolyData()
    vtpNew.DeepCopy(vtp)
    vtkPoints = vtpNew.GetPoints()

    # vtkPoints = vtk.vtkPoints()
    # vtkPoints.DeepCopy(basis.GetPoints())
    if len(coord) != vtp.GetNumberOfPoints(): 
        raise ValueError("number of coords must equal to number of points in vtp!")
    for i in range(vtpNew.GetNumberOfPoints()): 
        vtkPoints.SetPoint(i, coord[i, 0], coord[i, 1], coord[i, 2])

    vtpNew.SetPoints(vtkPoints)
    return vtpNew 

fitted_vtp = changeCoord(target_vtp, shape)
writePly(fitted_vtp, "fitted.ply")

# import pdb; pdb.set_trace()

############################# now begin batch processing #############################

def representAsARKit(basis, bsShapeArr, targetNeutral, target_vtp): 
    """return shape vtp that's represented by ARKit BS"""
    target = np.array(target_vtp.GetPoints().GetData())

    # x, residual, rank, singular_value = splitBSCoef(basis, bsShapeArr, targetNeutral, target)
    # x = x.clip(0, 1) # clip to [0, 1]
    # x = splitBSCoef_ridge(basis, bsShapeArr, targetNeutral, target)
    x = splitBSCoef_qp(basis, bsShapeArr, targetNeutral, target)
    x[x<1e-4] = 0

    A = bsShapeArr.T - np.repeat(basis.flatten().reshape(-1, 1), len(bsShapeArr), 1) # 15069x52

    delta = (A.T * x.reshape(-1, 1)).sum(axis=0).reshape(-1, 3) # 5023x3
    shape = targetNeutral + delta 

    newVtp = changeCoord(target_vtp, shape)
    return x, newVtp

inputID = '/home/xudong/data/unposedcleaneddata'
output = '/home/xudong/data/unposedcleaneddata_arkit'

def process_folder(inputID, output, id_name):
    inputFolder = os.path.join(inputID, id_name)
    outputFolder = os.path.join(output, id_name)

    tgt_neutral = os.path.join(inputFolder, 'sentence01/sentence01.000001.ply')
    targetNeutral_vtp = vtkReadMesh(tgt_neutral)
    targetNeutral = np.array(targetNeutral_vtp.GetPoints().GetData())

    sentence_names = [folder for folder in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, folder))]

    for sentence in sentence_names:
        process_sentence(inputFolder, outputFolder, sentence, targetNeutral)

def process_sentence(inputFolder, outputFolder, sentence, targetNeutral):
    xLs = []
    input_sentence_path = os.path.join(inputFolder, sentence)
    output_sentence_path = os.path.join(outputFolder, sentence)

    if not os.path.exists(output_sentence_path):
        os.makedirs(output_sentence_path)

    inputFileLs = sorted(glob.glob(os.path.join(input_sentence_path, '*.ply')))

    for input_file in inputFileLs:
        filename = os.path.basename(input_file)
        # print(f'representing as ARKit for file "{filename}"')

        output_file = os.path.join(output_sentence_path, filename[:-4] + '.ply')
        vtp = vtkReadMesh(input_file)
        x, newVtp = representAsARKit(basis, bsShapeArr, targetNeutral, vtp)
        xLs.append(x)
        writePly(newVtp, output_file)

    df = pd.DataFrame(columns=bsNameArr, data=np.array(xLs).squeeze())
    df.to_csv(os.path.join(output_sentence_path, '_arkit_coef.csv'), index=False)

folder_names = [folder for folder in os.listdir(inputID) if os.path.isdir(os.path.join(inputID, folder))]

for i, id_name in enumerate(folder_names):
    process_folder(inputID, output, id_name)
    print(f"finished {i+1}/{len(folder_names)}")

# folder_names = [folder for folder in os.listdir(inputID) if os.path.isdir(os.path.join(inputID, folder))]
#
# for i, id_name in enumerate(folder_names):
#
#     # tgt_neutral = '/home/xudong/data/VOCA_unposeddata/FaceTalk_170809_00138_TA/sentence01/sentence01.000001.ply'
#
#     inputFolder = os.path.join(inputID,id_name)
#     outputFolder = os.path.join(output,id_name)
#     # if not os.path.exists(outputFolder):
#     #     os.makedirs(outputFolder)
#
#     tgt_neutral = oj(inputFolder,'sentence01/sentence01.000001.ply')
#     targetNeutral_vtp = vtkReadMesh('FaceTalk_170904_03276_TA.ply')
#     targetNeutral = np.array(targetNeutral_vtp.GetPoints().GetData())
#     sentence_names = [folder for folder in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, folder))]
#     for s in sentence_names:
#         xLs = []
#         inputFileLs = sorted(glob.glob(oj(inputFolder, s,'*.ply')))
#         outputSentence = os.path.join(outputFolder,s)
#         if not os.path.exists(outputSentence):
#             os.makedirs(outputSentence)
#         for f in inputFileLs:
#             filename = os.path.split(f)[-1] # filename only
#             print('representing as ARKit for file "{}"'.format(filename))
#             outputFile = oj(outputSentence, filename[:-4]+'.ply')
#             vtp = vtkReadMesh(f)
#             x, newVtp = representAsARKit(basis, bsShapeArr, targetNeutral, vtp)
#             xLs.append(x)
#             writePly(newVtp, outputFile)
#         df = pd.DataFrame(columns=bsNameArr, data=np.array(xLs).squeeze())
#         df.to_csv(oj(outputSentence, '_arkit_coef.csv'), index=False)
#     print(f" finished {i}/{len(folder_names)}")
