import sys
import math

def ex_info():
    #Get info about the exception
	ex_type,ex_obj,ex_trace = sys.exc_info()
	line_no=ex_trace.tb_lineno
	print("\t Exception: ",ex_type)
	print("\t Line no.: ",line_no)

def distance2conf(face_distance,tolerance):
    # Calculate confidence based on the distance and tolerance
	if face_distance > tolerance:
		range = (1.0 - tolerance)
		linear_val = (1.0 - face_distance) / (range * 2.0)
		return linear_val
	else:
		range = tolerance
		linear_val = 1.0 - (face_distance / (range * 2.0))
		return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) *2,0.2))

