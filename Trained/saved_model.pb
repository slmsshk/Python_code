º
Ý­
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ÍÌL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ËÐ
¥
&Adam/input_layer_lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*7
shared_name(&Adam/input_layer_lstm/lstm_cell/bias/v

:Adam/input_layer_lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOp&Adam/input_layer_lstm/lstm_cell/bias/v*
_output_shapes	
:È*
dtype0
Á
2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2È*C
shared_name42Adam/input_layer_lstm/lstm_cell/recurrent_kernel/v
º
FAdam/input_layer_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/v*
_output_shapes
:	2È*
dtype0
­
(Adam/input_layer_lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È*9
shared_name*(Adam/input_layer_lstm/lstm_cell/kernel/v
¦
<Adam/input_layer_lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/input_layer_lstm/lstm_cell/kernel/v*
_output_shapes
:	È*
dtype0

Adam/Output_layer_Dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/Output_layer_Dense/bias/v

2Adam/Output_layer_Dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer_Dense/bias/v*
_output_shapes
:*
dtype0

 Adam/Output_layer_Dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*1
shared_name" Adam/Output_layer_Dense/kernel/v

4Adam/Output_layer_Dense/kernel/v/Read/ReadVariableOpReadVariableOp Adam/Output_layer_Dense/kernel/v*
_output_shapes

:2*
dtype0

Adam/Hidden_layer_Dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*0
shared_name!Adam/Hidden_layer_Dense2/bias/v

3Adam/Hidden_layer_Dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_Dense2/bias/v*
_output_shapes
:2*
dtype0

!Adam/Hidden_layer_Dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*2
shared_name#!Adam/Hidden_layer_Dense2/kernel/v

5Adam/Hidden_layer_Dense2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/Hidden_layer_Dense2/kernel/v*
_output_shapes

:22*
dtype0

Adam/Hidden_layer_Dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*0
shared_name!Adam/Hidden_layer_Dense1/bias/v

3Adam/Hidden_layer_Dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_Dense1/bias/v*
_output_shapes
:2*
dtype0

!Adam/Hidden_layer_Dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*2
shared_name#!Adam/Hidden_layer_Dense1/kernel/v

5Adam/Hidden_layer_Dense1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/Hidden_layer_Dense1/kernel/v*
_output_shapes

:22*
dtype0
¥
&Adam/input_layer_lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*7
shared_name(&Adam/input_layer_lstm/lstm_cell/bias/m

:Adam/input_layer_lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOp&Adam/input_layer_lstm/lstm_cell/bias/m*
_output_shapes	
:È*
dtype0
Á
2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2È*C
shared_name42Adam/input_layer_lstm/lstm_cell/recurrent_kernel/m
º
FAdam/input_layer_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/m*
_output_shapes
:	2È*
dtype0
­
(Adam/input_layer_lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È*9
shared_name*(Adam/input_layer_lstm/lstm_cell/kernel/m
¦
<Adam/input_layer_lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/input_layer_lstm/lstm_cell/kernel/m*
_output_shapes
:	È*
dtype0

Adam/Output_layer_Dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/Output_layer_Dense/bias/m

2Adam/Output_layer_Dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer_Dense/bias/m*
_output_shapes
:*
dtype0

 Adam/Output_layer_Dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*1
shared_name" Adam/Output_layer_Dense/kernel/m

4Adam/Output_layer_Dense/kernel/m/Read/ReadVariableOpReadVariableOp Adam/Output_layer_Dense/kernel/m*
_output_shapes

:2*
dtype0

Adam/Hidden_layer_Dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*0
shared_name!Adam/Hidden_layer_Dense2/bias/m

3Adam/Hidden_layer_Dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_Dense2/bias/m*
_output_shapes
:2*
dtype0

!Adam/Hidden_layer_Dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*2
shared_name#!Adam/Hidden_layer_Dense2/kernel/m

5Adam/Hidden_layer_Dense2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/Hidden_layer_Dense2/kernel/m*
_output_shapes

:22*
dtype0

Adam/Hidden_layer_Dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*0
shared_name!Adam/Hidden_layer_Dense1/bias/m

3Adam/Hidden_layer_Dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_Dense1/bias/m*
_output_shapes
:2*
dtype0

!Adam/Hidden_layer_Dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*2
shared_name#!Adam/Hidden_layer_Dense1/kernel/m

5Adam/Hidden_layer_Dense1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/Hidden_layer_Dense1/kernel/m*
_output_shapes

:22*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

input_layer_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*0
shared_name!input_layer_lstm/lstm_cell/bias

3input_layer_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpinput_layer_lstm/lstm_cell/bias*
_output_shapes	
:È*
dtype0
³
+input_layer_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2È*<
shared_name-+input_layer_lstm/lstm_cell/recurrent_kernel
¬
?input_layer_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp+input_layer_lstm/lstm_cell/recurrent_kernel*
_output_shapes
:	2È*
dtype0

!input_layer_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È*2
shared_name#!input_layer_lstm/lstm_cell/kernel

5input_layer_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp!input_layer_lstm/lstm_cell/kernel*
_output_shapes
:	È*
dtype0

Output_layer_Dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameOutput_layer_Dense/bias

+Output_layer_Dense/bias/Read/ReadVariableOpReadVariableOpOutput_layer_Dense/bias*
_output_shapes
:*
dtype0

Output_layer_Dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2**
shared_nameOutput_layer_Dense/kernel

-Output_layer_Dense/kernel/Read/ReadVariableOpReadVariableOpOutput_layer_Dense/kernel*
_output_shapes

:2*
dtype0

Hidden_layer_Dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameHidden_layer_Dense2/bias

,Hidden_layer_Dense2/bias/Read/ReadVariableOpReadVariableOpHidden_layer_Dense2/bias*
_output_shapes
:2*
dtype0

Hidden_layer_Dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*+
shared_nameHidden_layer_Dense2/kernel

.Hidden_layer_Dense2/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_Dense2/kernel*
_output_shapes

:22*
dtype0

Hidden_layer_Dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameHidden_layer_Dense1/bias

,Hidden_layer_Dense1/bias/Read/ReadVariableOpReadVariableOpHidden_layer_Dense1/bias*
_output_shapes
:2*
dtype0

Hidden_layer_Dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*+
shared_nameHidden_layer_Dense1/kernel

.Hidden_layer_Dense1/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_Dense1/kernel*
_output_shapes

:22*
dtype0

&serving_default_input_layer_lstm_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
ã
StatefulPartitionedCallStatefulPartitionedCall&serving_default_input_layer_lstm_input!input_layer_lstm/lstm_cell/kernel+input_layer_lstm/lstm_cell/recurrent_kernelinput_layer_lstm/lstm_cell/biasHidden_layer_Dense1/kernelHidden_layer_Dense1/biasHidden_layer_Dense2/kernelHidden_layer_Dense2/biasOutput_layer_Dense/kernelOutput_layer_Dense/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_18364

NoOpNoOp
G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÐF
valueÆFBÃF B¼F
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

activation

kernel
bias*
¦
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
¦
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
C
00
11
22
3
4
&5
'6
.7
/8*
C
00
11
22
3
4
&5
'6
.7
/8*
* 
°
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
8trace_0
9trace_1
:trace_2
;trace_3* 
6
<trace_0
=trace_1
>trace_2
?trace_3* 
* 
ø
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratemm&m'm.m/m0m1m2mvv&v'v.v/v0v1v2v*

Eserving_default* 

00
11
22*

00
11
22*
* 


Fstates
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
* 
ã
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator
[
state_size

0kernel
1recurrent_kernel
2bias*
* 

0
1*

0
1*
* 

\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 

c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses* 
jd
VARIABLE_VALUEHidden_layer_Dense1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEHidden_layer_Dense1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
jd
VARIABLE_VALUEHidden_layer_Dense2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEHidden_layer_Dense2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
ic
VARIABLE_VALUEOutput_layer_Dense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEOutput_layer_Dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!input_layer_lstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+input_layer_lstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEinput_layer_lstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

w0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

00
11
22*

00
11
22*
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

}trace_0
~trace_1* 

trace_0
trace_1* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/Hidden_layer_Dense1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Hidden_layer_Dense1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/Hidden_layer_Dense2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Hidden_layer_Dense2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/Output_layer_Dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Output_layer_Dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/input_layer_lstm/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/input_layer_lstm/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/Hidden_layer_Dense1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Hidden_layer_Dense1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/Hidden_layer_Dense2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Hidden_layer_Dense2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/Output_layer_Dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Output_layer_Dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/input_layer_lstm/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/input_layer_lstm/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
è
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.Hidden_layer_Dense1/kernel/Read/ReadVariableOp,Hidden_layer_Dense1/bias/Read/ReadVariableOp.Hidden_layer_Dense2/kernel/Read/ReadVariableOp,Hidden_layer_Dense2/bias/Read/ReadVariableOp-Output_layer_Dense/kernel/Read/ReadVariableOp+Output_layer_Dense/bias/Read/ReadVariableOp5input_layer_lstm/lstm_cell/kernel/Read/ReadVariableOp?input_layer_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp3input_layer_lstm/lstm_cell/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp5Adam/Hidden_layer_Dense1/kernel/m/Read/ReadVariableOp3Adam/Hidden_layer_Dense1/bias/m/Read/ReadVariableOp5Adam/Hidden_layer_Dense2/kernel/m/Read/ReadVariableOp3Adam/Hidden_layer_Dense2/bias/m/Read/ReadVariableOp4Adam/Output_layer_Dense/kernel/m/Read/ReadVariableOp2Adam/Output_layer_Dense/bias/m/Read/ReadVariableOp<Adam/input_layer_lstm/lstm_cell/kernel/m/Read/ReadVariableOpFAdam/input_layer_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp:Adam/input_layer_lstm/lstm_cell/bias/m/Read/ReadVariableOp5Adam/Hidden_layer_Dense1/kernel/v/Read/ReadVariableOp3Adam/Hidden_layer_Dense1/bias/v/Read/ReadVariableOp5Adam/Hidden_layer_Dense2/kernel/v/Read/ReadVariableOp3Adam/Hidden_layer_Dense2/bias/v/Read/ReadVariableOp4Adam/Output_layer_Dense/kernel/v/Read/ReadVariableOp2Adam/Output_layer_Dense/bias/v/Read/ReadVariableOp<Adam/input_layer_lstm/lstm_cell/kernel/v/Read/ReadVariableOpFAdam/input_layer_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp:Adam/input_layer_lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_19643
»

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden_layer_Dense1/kernelHidden_layer_Dense1/biasHidden_layer_Dense2/kernelHidden_layer_Dense2/biasOutput_layer_Dense/kernelOutput_layer_Dense/bias!input_layer_lstm/lstm_cell/kernel+input_layer_lstm/lstm_cell/recurrent_kernelinput_layer_lstm/lstm_cell/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount!Adam/Hidden_layer_Dense1/kernel/mAdam/Hidden_layer_Dense1/bias/m!Adam/Hidden_layer_Dense2/kernel/mAdam/Hidden_layer_Dense2/bias/m Adam/Output_layer_Dense/kernel/mAdam/Output_layer_Dense/bias/m(Adam/input_layer_lstm/lstm_cell/kernel/m2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/m&Adam/input_layer_lstm/lstm_cell/bias/m!Adam/Hidden_layer_Dense1/kernel/vAdam/Hidden_layer_Dense1/bias/v!Adam/Hidden_layer_Dense2/kernel/vAdam/Hidden_layer_Dense2/bias/v Adam/Output_layer_Dense/kernel/vAdam/Output_layer_Dense/bias/v(Adam/input_layer_lstm/lstm_cell/kernel/v2Adam/input_layer_lstm/lstm_cell/recurrent_kernel/v&Adam/input_layer_lstm/lstm_cell/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_19755¥
À\

/Sequence_LSTM_input_layer_lstm_while_body_17301Z
Vsequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_loop_counter`
\sequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_maximum_iterations4
0sequence_lstm_input_layer_lstm_while_placeholder6
2sequence_lstm_input_layer_lstm_while_placeholder_16
2sequence_lstm_input_layer_lstm_while_placeholder_26
2sequence_lstm_input_layer_lstm_while_placeholder_3Y
Usequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_strided_slice_1_0
sequence_lstm_input_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_sequence_lstm_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0b
Osequence_lstm_input_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0:	Èd
Qsequence_lstm_input_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:	2È_
Psequence_lstm_input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:	È1
-sequence_lstm_input_layer_lstm_while_identity3
/sequence_lstm_input_layer_lstm_while_identity_13
/sequence_lstm_input_layer_lstm_while_identity_23
/sequence_lstm_input_layer_lstm_while_identity_33
/sequence_lstm_input_layer_lstm_while_identity_43
/sequence_lstm_input_layer_lstm_while_identity_5W
Ssequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_strided_slice_1
sequence_lstm_input_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_sequence_lstm_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor`
Msequence_lstm_input_layer_lstm_while_lstm_cell_matmul_readvariableop_resource:	Èb
Osequence_lstm_input_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource:	2È]
Nsequence_lstm_input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource:	È¢ESequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢DSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp¢FSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp§
VSequence_LSTM/input_layer_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
HSequence_LSTM/input_layer_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequence_lstm_input_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_sequence_lstm_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_00sequence_lstm_input_layer_lstm_while_placeholder_Sequence_LSTM/input_layer_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Õ
DSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpOsequence_lstm_input_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0
5Sequence_LSTM/input_layer_lstm/while/lstm_cell/MatMulMatMulOSequence_LSTM/input_layer_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0LSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÙ
FSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpQsequence_lstm_input_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0ø
7Sequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1MatMul2sequence_lstm_input_layer_lstm_while_placeholder_2NSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈò
2Sequence_LSTM/input_layer_lstm/while/lstm_cell/addAddV2?Sequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul:product:0ASequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÓ
ESequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpPsequence_lstm_input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0û
6Sequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAddBiasAdd6Sequence_LSTM/input_layer_lstm/while/lstm_cell/add:z:0MSequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
>Sequence_LSTM/input_layer_lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ã
4Sequence_LSTM/input_layer_lstm/while/lstm_cell/splitSplitGSequence_LSTM/input_layer_lstm/while/lstm_cell/split/split_dim:output:0?Sequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split²
6Sequence_LSTM/input_layer_lstm/while/lstm_cell/SigmoidSigmoid=Sequence_LSTM/input_layer_lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2´
8Sequence_LSTM/input_layer_lstm/while/lstm_cell/Sigmoid_1Sigmoid=Sequence_LSTM/input_layer_lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ý
2Sequence_LSTM/input_layer_lstm/while/lstm_cell/mulMul<Sequence_LSTM/input_layer_lstm/while/lstm_cell/Sigmoid_1:y:02sequence_lstm_input_layer_lstm_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
3Sequence_LSTM/input_layer_lstm/while/lstm_cell/ReluRelu=Sequence_LSTM/input_layer_lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ì
4Sequence_LSTM/input_layer_lstm/while/lstm_cell/mul_1Mul:Sequence_LSTM/input_layer_lstm/while/lstm_cell/Sigmoid:y:0ASequence_LSTM/input_layer_lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2á
4Sequence_LSTM/input_layer_lstm/while/lstm_cell/add_1AddV26Sequence_LSTM/input_layer_lstm/while/lstm_cell/mul:z:08Sequence_LSTM/input_layer_lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2´
8Sequence_LSTM/input_layer_lstm/while/lstm_cell/Sigmoid_2Sigmoid=Sequence_LSTM/input_layer_lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2©
5Sequence_LSTM/input_layer_lstm/while/lstm_cell/Relu_1Relu8Sequence_LSTM/input_layer_lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ð
4Sequence_LSTM/input_layer_lstm/while/lstm_cell/mul_2Mul<Sequence_LSTM/input_layer_lstm/while/lstm_cell/Sigmoid_2:y:0CSequence_LSTM/input_layer_lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OSequence_LSTM/input_layer_lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : æ
ISequence_LSTM/input_layer_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2sequence_lstm_input_layer_lstm_while_placeholder_1XSequence_LSTM/input_layer_lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:08Sequence_LSTM/input_layer_lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒl
*Sequence_LSTM/input_layer_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(Sequence_LSTM/input_layer_lstm/while/addAddV20sequence_lstm_input_layer_lstm_while_placeholder3Sequence_LSTM/input_layer_lstm/while/add/y:output:0*
T0*
_output_shapes
: n
,Sequence_LSTM/input_layer_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*Sequence_LSTM/input_layer_lstm/while/add_1AddV2Vsequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_loop_counter5Sequence_LSTM/input_layer_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ¶
-Sequence_LSTM/input_layer_lstm/while/IdentityIdentity.Sequence_LSTM/input_layer_lstm/while/add_1:z:0*^Sequence_LSTM/input_layer_lstm/while/NoOp*
T0*
_output_shapes
: æ
/Sequence_LSTM/input_layer_lstm/while/Identity_1Identity\sequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_maximum_iterations*^Sequence_LSTM/input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ¶
/Sequence_LSTM/input_layer_lstm/while/Identity_2Identity,Sequence_LSTM/input_layer_lstm/while/add:z:0*^Sequence_LSTM/input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ã
/Sequence_LSTM/input_layer_lstm/while/Identity_3IdentityYSequence_LSTM/input_layer_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^Sequence_LSTM/input_layer_lstm/while/NoOp*
T0*
_output_shapes
: Ó
/Sequence_LSTM/input_layer_lstm/while/Identity_4Identity8Sequence_LSTM/input_layer_lstm/while/lstm_cell/mul_2:z:0*^Sequence_LSTM/input_layer_lstm/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ó
/Sequence_LSTM/input_layer_lstm/while/Identity_5Identity8Sequence_LSTM/input_layer_lstm/while/lstm_cell/add_1:z:0*^Sequence_LSTM/input_layer_lstm/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ã
)Sequence_LSTM/input_layer_lstm/while/NoOpNoOpF^Sequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOpE^Sequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOpG^Sequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "g
-sequence_lstm_input_layer_lstm_while_identity6Sequence_LSTM/input_layer_lstm/while/Identity:output:0"k
/sequence_lstm_input_layer_lstm_while_identity_18Sequence_LSTM/input_layer_lstm/while/Identity_1:output:0"k
/sequence_lstm_input_layer_lstm_while_identity_28Sequence_LSTM/input_layer_lstm/while/Identity_2:output:0"k
/sequence_lstm_input_layer_lstm_while_identity_38Sequence_LSTM/input_layer_lstm/while/Identity_3:output:0"k
/sequence_lstm_input_layer_lstm_while_identity_48Sequence_LSTM/input_layer_lstm/while/Identity_4:output:0"k
/sequence_lstm_input_layer_lstm_while_identity_58Sequence_LSTM/input_layer_lstm/while/Identity_5:output:0"¢
Nsequence_lstm_input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resourcePsequence_lstm_input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"¤
Osequence_lstm_input_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resourceQsequence_lstm_input_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0" 
Msequence_lstm_input_layer_lstm_while_lstm_cell_matmul_readvariableop_resourceOsequence_lstm_input_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0"¬
Ssequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_strided_slice_1Usequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_strided_slice_1_0"¦
sequence_lstm_input_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_sequence_lstm_input_layer_lstm_tensorarrayunstack_tensorlistfromtensorsequence_lstm_input_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_sequence_lstm_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2
ESequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOpESequence_LSTM/input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp2
DSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOpDSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp2
FSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOpFSequence_LSTM/input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
â

H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_17968

inputs)
input_layer_lstm_17911:	È)
input_layer_lstm_17913:	2È%
input_layer_lstm_17915:	È+
hidden_layer_dense1_17930:22'
hidden_layer_dense1_17932:2+
hidden_layer_dense2_17946:22'
hidden_layer_dense2_17948:2*
output_layer_dense_17962:2&
output_layer_dense_17964:
identity¢+Hidden_layer_Dense1/StatefulPartitionedCall¢+Hidden_layer_Dense2/StatefulPartitionedCall¢*Output_layer_Dense/StatefulPartitionedCall¢(input_layer_lstm/StatefulPartitionedCall§
(input_layer_lstm/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_lstm_17911input_layer_lstm_17913input_layer_lstm_17915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17910Ä
+Hidden_layer_Dense1/StatefulPartitionedCallStatefulPartitionedCall1input_layer_lstm/StatefulPartitionedCall:output:0hidden_layer_dense1_17930hidden_layer_dense1_17932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_17929Ç
+Hidden_layer_Dense2/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense1/StatefulPartitionedCall:output:0hidden_layer_dense2_17946hidden_layer_dense2_17948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_17945Ã
*Output_layer_Dense/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense2/StatefulPartitionedCall:output:0output_layer_dense_17962output_layer_dense_17964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_17961
IdentityIdentity3Output_layer_Dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp,^Hidden_layer_Dense1/StatefulPartitionedCall,^Hidden_layer_Dense2/StatefulPartitionedCall+^Output_layer_Dense/StatefulPartitionedCall)^input_layer_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2Z
+Hidden_layer_Dense1/StatefulPartitionedCall+Hidden_layer_Dense1/StatefulPartitionedCall2Z
+Hidden_layer_Dense2/StatefulPartitionedCall+Hidden_layer_Dense2/StatefulPartitionedCall2X
*Output_layer_Dense/StatefulPartitionedCall*Output_layer_Dense/StatefulPartitionedCall2T
(input_layer_lstm/StatefulPartitionedCall(input_layer_lstm/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ýI

K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19362

inputs;
(lstm_cell_matmul_readvariableop_resource:	È=
*lstm_cell_matmul_1_readvariableop_resource:	2È8
)lstm_cell_biasadd_readvariableop_resource:	È
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ô
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19277*
condR
while_cond_19276*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


!input_layer_lstm_while_cond_18469>
:input_layer_lstm_while_input_layer_lstm_while_loop_counterD
@input_layer_lstm_while_input_layer_lstm_while_maximum_iterations&
"input_layer_lstm_while_placeholder(
$input_layer_lstm_while_placeholder_1(
$input_layer_lstm_while_placeholder_2(
$input_layer_lstm_while_placeholder_3@
<input_layer_lstm_while_less_input_layer_lstm_strided_slice_1U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18469___redundant_placeholder0U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18469___redundant_placeholder1U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18469___redundant_placeholder2U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18469___redundant_placeholder3#
input_layer_lstm_while_identity
¦
input_layer_lstm/while/LessLess"input_layer_lstm_while_placeholder<input_layer_lstm_while_less_input_layer_lstm_strided_slice_1*
T0*
_output_shapes
: m
input_layer_lstm/while/IdentityIdentityinput_layer_lstm/while/Less:z:0*
T0
*
_output_shapes
: "K
input_layer_lstm_while_identity(input_layer_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
Û7
´
while_body_18842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ÈE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	2È@
1while_lstm_cell_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ÈC
0while_lstm_cell_matmul_1_readvariableop_resource:	2È>
/while_lstm_cell_biasadd_readvariableop_resource:	È¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :æ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ç

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ç{
§	
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18738

inputsL
9input_layer_lstm_lstm_cell_matmul_readvariableop_resource:	ÈN
;input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource:	2ÈI
:input_layer_lstm_lstm_cell_biasadd_readvariableop_resource:	ÈD
2hidden_layer_dense1_matmul_readvariableop_resource:22A
3hidden_layer_dense1_biasadd_readvariableop_resource:2D
2hidden_layer_dense2_matmul_readvariableop_resource:22A
3hidden_layer_dense2_biasadd_readvariableop_resource:2C
1output_layer_dense_matmul_readvariableop_resource:2@
2output_layer_dense_biasadd_readvariableop_resource:
identity¢*Hidden_layer_Dense1/BiasAdd/ReadVariableOp¢)Hidden_layer_Dense1/MatMul/ReadVariableOp¢*Hidden_layer_Dense2/BiasAdd/ReadVariableOp¢)Hidden_layer_Dense2/MatMul/ReadVariableOp¢)Output_layer_Dense/BiasAdd/ReadVariableOp¢(Output_layer_Dense/MatMul/ReadVariableOp¢1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp¢0input_layer_lstm/lstm_cell/MatMul/ReadVariableOp¢2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp¢input_layer_lstm/whileL
input_layer_lstm/ShapeShapeinputs*
T0*
_output_shapes
:n
$input_layer_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&input_layer_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&input_layer_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
input_layer_lstm/strided_sliceStridedSliceinput_layer_lstm/Shape:output:0-input_layer_lstm/strided_slice/stack:output:0/input_layer_lstm/strided_slice/stack_1:output:0/input_layer_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
input_layer_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2¦
input_layer_lstm/zeros/packedPack'input_layer_lstm/strided_slice:output:0(input_layer_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
input_layer_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
input_layer_lstm/zerosFill&input_layer_lstm/zeros/packed:output:0%input_layer_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
!input_layer_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2ª
input_layer_lstm/zeros_1/packedPack'input_layer_lstm/strided_slice:output:0*input_layer_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
input_layer_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
input_layer_lstm/zeros_1Fill(input_layer_lstm/zeros_1/packed:output:0'input_layer_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
input_layer_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
input_layer_lstm/transpose	Transposeinputs(input_layer_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
input_layer_lstm/Shape_1Shapeinput_layer_lstm/transpose:y:0*
T0*
_output_shapes
:p
&input_layer_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(input_layer_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(input_layer_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 input_layer_lstm/strided_slice_1StridedSlice!input_layer_lstm/Shape_1:output:0/input_layer_lstm/strided_slice_1/stack:output:01input_layer_lstm/strided_slice_1/stack_1:output:01input_layer_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,input_layer_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿç
input_layer_lstm/TensorArrayV2TensorListReserve5input_layer_lstm/TensorArrayV2/element_shape:output:0)input_layer_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Finput_layer_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8input_layer_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinput_layer_lstm/transpose:y:0Oinput_layer_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒp
&input_layer_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(input_layer_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(input_layer_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
 input_layer_lstm/strided_slice_2StridedSliceinput_layer_lstm/transpose:y:0/input_layer_lstm/strided_slice_2/stack:output:01input_layer_lstm/strided_slice_2/stack_1:output:01input_layer_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask«
0input_layer_lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp9input_layer_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0Ã
!input_layer_lstm/lstm_cell/MatMulMatMul)input_layer_lstm/strided_slice_2:output:08input_layer_lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¯
2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0½
#input_layer_lstm/lstm_cell/MatMul_1MatMulinput_layer_lstm/zeros:output:0:input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¶
input_layer_lstm/lstm_cell/addAddV2+input_layer_lstm/lstm_cell/MatMul:product:0-input_layer_lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:input_layer_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0¿
"input_layer_lstm/lstm_cell/BiasAddBiasAdd"input_layer_lstm/lstm_cell/add:z:09input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
*input_layer_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 input_layer_lstm/lstm_cell/splitSplit3input_layer_lstm/lstm_cell/split/split_dim:output:0+input_layer_lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
"input_layer_lstm/lstm_cell/SigmoidSigmoid)input_layer_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
$input_layer_lstm/lstm_cell/Sigmoid_1Sigmoid)input_layer_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¤
input_layer_lstm/lstm_cell/mulMul(input_layer_lstm/lstm_cell/Sigmoid_1:y:0!input_layer_lstm/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
input_layer_lstm/lstm_cell/ReluRelu)input_layer_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2°
 input_layer_lstm/lstm_cell/mul_1Mul&input_layer_lstm/lstm_cell/Sigmoid:y:0-input_layer_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¥
 input_layer_lstm/lstm_cell/add_1AddV2"input_layer_lstm/lstm_cell/mul:z:0$input_layer_lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
$input_layer_lstm/lstm_cell/Sigmoid_2Sigmoid)input_layer_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
!input_layer_lstm/lstm_cell/Relu_1Relu$input_layer_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2´
 input_layer_lstm/lstm_cell/mul_2Mul(input_layer_lstm/lstm_cell/Sigmoid_2:y:0/input_layer_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
.input_layer_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   o
-input_layer_lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ø
 input_layer_lstm/TensorArrayV2_1TensorListReserve7input_layer_lstm/TensorArrayV2_1/element_shape:output:06input_layer_lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒW
input_layer_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)input_layer_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
#input_layer_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å
input_layer_lstm/whileWhile,input_layer_lstm/while/loop_counter:output:02input_layer_lstm/while/maximum_iterations:output:0input_layer_lstm/time:output:0)input_layer_lstm/TensorArrayV2_1:handle:0input_layer_lstm/zeros:output:0!input_layer_lstm/zeros_1:output:0)input_layer_lstm/strided_slice_1:output:0Hinput_layer_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:09input_layer_lstm_lstm_cell_matmul_readvariableop_resource;input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource:input_layer_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!input_layer_lstm_while_body_18634*-
cond%R#
!input_layer_lstm_while_cond_18633*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
Ainput_layer_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   
3input_layer_lstm/TensorArrayV2Stack/TensorListStackTensorListStackinput_layer_lstm/while:output:3Jinput_layer_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsy
&input_layer_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿr
(input_layer_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(input_layer_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
 input_layer_lstm/strided_slice_3StridedSlice<input_layer_lstm/TensorArrayV2Stack/TensorListStack:tensor:0/input_layer_lstm/strided_slice_3/stack:output:01input_layer_lstm/strided_slice_3/stack_1:output:01input_layer_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskv
!input_layer_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
input_layer_lstm/transpose_1	Transpose<input_layer_lstm/TensorArrayV2Stack/TensorListStack:tensor:0*input_layer_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2l
input_layer_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
)Hidden_layer_Dense1/MatMul/ReadVariableOpReadVariableOp2hidden_layer_dense1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0´
Hidden_layer_Dense1/MatMulMatMul)input_layer_lstm/strided_slice_3:output:01Hidden_layer_Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*Hidden_layer_Dense1/BiasAdd/ReadVariableOpReadVariableOp3hidden_layer_dense1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0²
Hidden_layer_Dense1/BiasAddBiasAdd$Hidden_layer_Dense1/MatMul:product:02Hidden_layer_Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
)Hidden_layer_Dense1/leaky_re_lu/LeakyRelu	LeakyRelu$Hidden_layer_Dense1/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
alpha%>
)Hidden_layer_Dense2/MatMul/ReadVariableOpReadVariableOp2hidden_layer_dense2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0Â
Hidden_layer_Dense2/MatMulMatMul7Hidden_layer_Dense1/leaky_re_lu/LeakyRelu:activations:01Hidden_layer_Dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*Hidden_layer_Dense2/BiasAdd/ReadVariableOpReadVariableOp3hidden_layer_dense2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0²
Hidden_layer_Dense2/BiasAddBiasAdd$Hidden_layer_Dense2/MatMul:product:02Hidden_layer_Dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
(Output_layer_Dense/MatMul/ReadVariableOpReadVariableOp1output_layer_dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0­
Output_layer_Dense/MatMulMatMul$Hidden_layer_Dense2/BiasAdd:output:00Output_layer_Dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)Output_layer_Dense/BiasAdd/ReadVariableOpReadVariableOp2output_layer_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
Output_layer_Dense/BiasAddBiasAdd#Output_layer_Dense/MatMul:product:01Output_layer_Dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity#Output_layer_Dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp+^Hidden_layer_Dense1/BiasAdd/ReadVariableOp*^Hidden_layer_Dense1/MatMul/ReadVariableOp+^Hidden_layer_Dense2/BiasAdd/ReadVariableOp*^Hidden_layer_Dense2/MatMul/ReadVariableOp*^Output_layer_Dense/BiasAdd/ReadVariableOp)^Output_layer_Dense/MatMul/ReadVariableOp2^input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp1^input_layer_lstm/lstm_cell/MatMul/ReadVariableOp3^input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp^input_layer_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2X
*Hidden_layer_Dense1/BiasAdd/ReadVariableOp*Hidden_layer_Dense1/BiasAdd/ReadVariableOp2V
)Hidden_layer_Dense1/MatMul/ReadVariableOp)Hidden_layer_Dense1/MatMul/ReadVariableOp2X
*Hidden_layer_Dense2/BiasAdd/ReadVariableOp*Hidden_layer_Dense2/BiasAdd/ReadVariableOp2V
)Hidden_layer_Dense2/MatMul/ReadVariableOp)Hidden_layer_Dense2/MatMul/ReadVariableOp2V
)Output_layer_Dense/BiasAdd/ReadVariableOp)Output_layer_Dense/BiasAdd/ReadVariableOp2T
(Output_layer_Dense/MatMul/ReadVariableOp(Output_layer_Dense/MatMul/ReadVariableOp2f
1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp2d
0input_layer_lstm/lstm_cell/MatMul/ReadVariableOp0input_layer_lstm/lstm_cell/MatMul/ReadVariableOp2h
2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp20
input_layer_lstm/whileinput_layer_lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
òK
Ô
!input_layer_lstm_while_body_18470>
:input_layer_lstm_while_input_layer_lstm_while_loop_counterD
@input_layer_lstm_while_input_layer_lstm_while_maximum_iterations&
"input_layer_lstm_while_placeholder(
$input_layer_lstm_while_placeholder_1(
$input_layer_lstm_while_placeholder_2(
$input_layer_lstm_while_placeholder_3=
9input_layer_lstm_while_input_layer_lstm_strided_slice_1_0y
uinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0T
Ainput_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0:	ÈV
Cinput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:	2ÈQ
Binput_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:	È#
input_layer_lstm_while_identity%
!input_layer_lstm_while_identity_1%
!input_layer_lstm_while_identity_2%
!input_layer_lstm_while_identity_3%
!input_layer_lstm_while_identity_4%
!input_layer_lstm_while_identity_5;
7input_layer_lstm_while_input_layer_lstm_strided_slice_1w
sinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensorR
?input_layer_lstm_while_lstm_cell_matmul_readvariableop_resource:	ÈT
Ainput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource:	2ÈO
@input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource:	È¢7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp¢8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp
Hinput_layer_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   û
:input_layer_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0"input_layer_lstm_while_placeholderQinput_layer_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¹
6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAinput_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0ç
'input_layer_lstm/while/lstm_cell/MatMulMatMulAinput_layer_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0>input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ½
8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCinput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0Î
)input_layer_lstm/while/lstm_cell/MatMul_1MatMul$input_layer_lstm_while_placeholder_2@input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
$input_layer_lstm/while/lstm_cell/addAddV21input_layer_lstm/while/lstm_cell/MatMul:product:03input_layer_lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ·
7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBinput_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0Ñ
(input_layer_lstm/while/lstm_cell/BiasAddBiasAdd(input_layer_lstm/while/lstm_cell/add:z:0?input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈr
0input_layer_lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&input_layer_lstm/while/lstm_cell/splitSplit9input_layer_lstm/while/lstm_cell/split/split_dim:output:01input_layer_lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
(input_layer_lstm/while/lstm_cell/SigmoidSigmoid/input_layer_lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*input_layer_lstm/while/lstm_cell/Sigmoid_1Sigmoid/input_layer_lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2³
$input_layer_lstm/while/lstm_cell/mulMul.input_layer_lstm/while/lstm_cell/Sigmoid_1:y:0$input_layer_lstm_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
%input_layer_lstm/while/lstm_cell/ReluRelu/input_layer_lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Â
&input_layer_lstm/while/lstm_cell/mul_1Mul,input_layer_lstm/while/lstm_cell/Sigmoid:y:03input_layer_lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
&input_layer_lstm/while/lstm_cell/add_1AddV2(input_layer_lstm/while/lstm_cell/mul:z:0*input_layer_lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*input_layer_lstm/while/lstm_cell/Sigmoid_2Sigmoid/input_layer_lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'input_layer_lstm/while/lstm_cell/Relu_1Relu*input_layer_lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Æ
&input_layer_lstm/while/lstm_cell/mul_2Mul.input_layer_lstm/while/lstm_cell/Sigmoid_2:y:05input_layer_lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Ainput_layer_lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ®
;input_layer_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$input_layer_lstm_while_placeholder_1Jinput_layer_lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0*input_layer_lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ^
input_layer_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
input_layer_lstm/while/addAddV2"input_layer_lstm_while_placeholder%input_layer_lstm/while/add/y:output:0*
T0*
_output_shapes
: `
input_layer_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :«
input_layer_lstm/while/add_1AddV2:input_layer_lstm_while_input_layer_lstm_while_loop_counter'input_layer_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
input_layer_lstm/while/IdentityIdentity input_layer_lstm/while/add_1:z:0^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ®
!input_layer_lstm/while/Identity_1Identity@input_layer_lstm_while_input_layer_lstm_while_maximum_iterations^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: 
!input_layer_lstm/while/Identity_2Identityinput_layer_lstm/while/add:z:0^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ¹
!input_layer_lstm/while/Identity_3IdentityKinput_layer_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ©
!input_layer_lstm/while/Identity_4Identity*input_layer_lstm/while/lstm_cell/mul_2:z:0^input_layer_lstm/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2©
!input_layer_lstm/while/Identity_5Identity*input_layer_lstm/while/lstm_cell/add_1:z:0^input_layer_lstm/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
input_layer_lstm/while/NoOpNoOp8^input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp7^input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp9^input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "K
input_layer_lstm_while_identity(input_layer_lstm/while/Identity:output:0"O
!input_layer_lstm_while_identity_1*input_layer_lstm/while/Identity_1:output:0"O
!input_layer_lstm_while_identity_2*input_layer_lstm/while/Identity_2:output:0"O
!input_layer_lstm_while_identity_3*input_layer_lstm/while/Identity_3:output:0"O
!input_layer_lstm_while_identity_4*input_layer_lstm/while/Identity_4:output:0"O
!input_layer_lstm_while_identity_5*input_layer_lstm/while/Identity_5:output:0"t
7input_layer_lstm_while_input_layer_lstm_strided_slice_19input_layer_lstm_while_input_layer_lstm_strided_slice_1_0"
@input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resourceBinput_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
Ainput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resourceCinput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"
?input_layer_lstm_while_lstm_cell_matmul_readvariableop_resourceAinput_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0"ì
sinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensoruinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2r
7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp2p
6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp2t
8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ýI

K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19217

inputs;
(lstm_cell_matmul_readvariableop_resource:	È=
*lstm_cell_matmul_1_readvariableop_resource:	2È8
)lstm_cell_biasadd_readvariableop_resource:	È
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ô
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19132*
condR
while_cond_19131*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ	
ÿ
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_19401

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ö
 
3__inference_Hidden_layer_Dense1_layer_call_fn_19371

inputs
unknown:22
	unknown_0:2
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_17929o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ñ

D__inference_lstm_cell_layer_call_and_return_conditional_losses_17620

inputs

states
states_11
matmul_readvariableop_resource:	È3
 matmul_1_readvariableop_resource:	2È.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates

½
0__inference_input_layer_lstm_layer_call_fn_18782

inputs
unknown:	È
	unknown_0:	2È
	unknown_1:	È
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ýI

K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18177

inputs;
(lstm_cell_matmul_readvariableop_resource:	È=
*lstm_cell_matmul_1_readvariableop_resource:	2È8
)lstm_cell_biasadd_readvariableop_resource:	È
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ô
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_18092*
condR
while_cond_18091*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¨
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18307
input_layer_lstm_input)
input_layer_lstm_18284:	È)
input_layer_lstm_18286:	2È%
input_layer_lstm_18288:	È+
hidden_layer_dense1_18291:22'
hidden_layer_dense1_18293:2+
hidden_layer_dense2_18296:22'
hidden_layer_dense2_18298:2*
output_layer_dense_18301:2&
output_layer_dense_18303:
identity¢+Hidden_layer_Dense1/StatefulPartitionedCall¢+Hidden_layer_Dense2/StatefulPartitionedCall¢*Output_layer_Dense/StatefulPartitionedCall¢(input_layer_lstm/StatefulPartitionedCall·
(input_layer_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_layer_lstm_inputinput_layer_lstm_18284input_layer_lstm_18286input_layer_lstm_18288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17910Ä
+Hidden_layer_Dense1/StatefulPartitionedCallStatefulPartitionedCall1input_layer_lstm/StatefulPartitionedCall:output:0hidden_layer_dense1_18291hidden_layer_dense1_18293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_17929Ç
+Hidden_layer_Dense2/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense1/StatefulPartitionedCall:output:0hidden_layer_dense2_18296hidden_layer_dense2_18298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_17945Ã
*Output_layer_Dense/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense2/StatefulPartitionedCall:output:0output_layer_dense_18301output_layer_dense_18303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_17961
IdentityIdentity3Output_layer_Dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp,^Hidden_layer_Dense1/StatefulPartitionedCall,^Hidden_layer_Dense2/StatefulPartitionedCall+^Output_layer_Dense/StatefulPartitionedCall)^input_layer_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2Z
+Hidden_layer_Dense1/StatefulPartitionedCall+Hidden_layer_Dense1/StatefulPartitionedCall2Z
+Hidden_layer_Dense2/StatefulPartitionedCall+Hidden_layer_Dense2/StatefulPartitionedCall2X
*Output_layer_Dense/StatefulPartitionedCall*Output_layer_Dense/StatefulPartitionedCall2T
(input_layer_lstm/StatefulPartitionedCall(input_layer_lstm/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_nameinput_layer_lstm_input

¿
0__inference_input_layer_lstm_layer_call_fn_18749
inputs_0
unknown:	È
	unknown_0:	2È
	unknown_1:	È
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
û	
â
#__inference_signature_wrapper_18364
input_layer_lstm_input
unknown:	È
	unknown_0:	2È
	unknown_1:	È
	unknown_2:22
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_layer_lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_17405o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_nameinput_layer_lstm_input
©#
Ç
while_body_17487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_17511_0:	È*
while_lstm_cell_17513_0:	2È&
while_lstm_cell_17515_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_17511:	È(
while_lstm_cell_17513:	2È$
while_lstm_cell_17515:	È¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0 
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_17511_0while_lstm_cell_17513_0while_lstm_cell_17515_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17472r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_17511while_lstm_cell_17511_0"0
while_lstm_cell_17513while_lstm_cell_17513_0"0
while_lstm_cell_17515while_lstm_cell_17515_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Û7
´
while_body_19277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ÈE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	2È@
1while_lstm_cell_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ÈC
0while_lstm_cell_matmul_1_readvariableop_resource:	2È>
/while_lstm_cell_biasadd_readvariableop_resource:	È¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :æ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ç

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ì8
þ
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17750

inputs"
lstm_cell_17666:	È"
lstm_cell_17668:	2È
lstm_cell_17670:	È
identity¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskâ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_17666lstm_cell_17668lstm_cell_17670*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17620n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_17666lstm_cell_17668lstm_cell_17670*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_17680*
condR
while_cond_17679*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û7
´
while_body_18987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ÈE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	2È@
1while_lstm_cell_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ÈC
0while_lstm_cell_matmul_1_readvariableop_resource:	2È>
/while_lstm_cell_biasadd_readvariableop_resource:	È¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :æ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ç

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ç{
§	
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18574

inputsL
9input_layer_lstm_lstm_cell_matmul_readvariableop_resource:	ÈN
;input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource:	2ÈI
:input_layer_lstm_lstm_cell_biasadd_readvariableop_resource:	ÈD
2hidden_layer_dense1_matmul_readvariableop_resource:22A
3hidden_layer_dense1_biasadd_readvariableop_resource:2D
2hidden_layer_dense2_matmul_readvariableop_resource:22A
3hidden_layer_dense2_biasadd_readvariableop_resource:2C
1output_layer_dense_matmul_readvariableop_resource:2@
2output_layer_dense_biasadd_readvariableop_resource:
identity¢*Hidden_layer_Dense1/BiasAdd/ReadVariableOp¢)Hidden_layer_Dense1/MatMul/ReadVariableOp¢*Hidden_layer_Dense2/BiasAdd/ReadVariableOp¢)Hidden_layer_Dense2/MatMul/ReadVariableOp¢)Output_layer_Dense/BiasAdd/ReadVariableOp¢(Output_layer_Dense/MatMul/ReadVariableOp¢1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp¢0input_layer_lstm/lstm_cell/MatMul/ReadVariableOp¢2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp¢input_layer_lstm/whileL
input_layer_lstm/ShapeShapeinputs*
T0*
_output_shapes
:n
$input_layer_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&input_layer_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&input_layer_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
input_layer_lstm/strided_sliceStridedSliceinput_layer_lstm/Shape:output:0-input_layer_lstm/strided_slice/stack:output:0/input_layer_lstm/strided_slice/stack_1:output:0/input_layer_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
input_layer_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2¦
input_layer_lstm/zeros/packedPack'input_layer_lstm/strided_slice:output:0(input_layer_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
input_layer_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
input_layer_lstm/zerosFill&input_layer_lstm/zeros/packed:output:0%input_layer_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
!input_layer_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2ª
input_layer_lstm/zeros_1/packedPack'input_layer_lstm/strided_slice:output:0*input_layer_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
input_layer_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
input_layer_lstm/zeros_1Fill(input_layer_lstm/zeros_1/packed:output:0'input_layer_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
input_layer_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
input_layer_lstm/transpose	Transposeinputs(input_layer_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
input_layer_lstm/Shape_1Shapeinput_layer_lstm/transpose:y:0*
T0*
_output_shapes
:p
&input_layer_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(input_layer_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(input_layer_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 input_layer_lstm/strided_slice_1StridedSlice!input_layer_lstm/Shape_1:output:0/input_layer_lstm/strided_slice_1/stack:output:01input_layer_lstm/strided_slice_1/stack_1:output:01input_layer_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,input_layer_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿç
input_layer_lstm/TensorArrayV2TensorListReserve5input_layer_lstm/TensorArrayV2/element_shape:output:0)input_layer_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Finput_layer_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8input_layer_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinput_layer_lstm/transpose:y:0Oinput_layer_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒp
&input_layer_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(input_layer_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(input_layer_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
 input_layer_lstm/strided_slice_2StridedSliceinput_layer_lstm/transpose:y:0/input_layer_lstm/strided_slice_2/stack:output:01input_layer_lstm/strided_slice_2/stack_1:output:01input_layer_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask«
0input_layer_lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp9input_layer_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0Ã
!input_layer_lstm/lstm_cell/MatMulMatMul)input_layer_lstm/strided_slice_2:output:08input_layer_lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¯
2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0½
#input_layer_lstm/lstm_cell/MatMul_1MatMulinput_layer_lstm/zeros:output:0:input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¶
input_layer_lstm/lstm_cell/addAddV2+input_layer_lstm/lstm_cell/MatMul:product:0-input_layer_lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:input_layer_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0¿
"input_layer_lstm/lstm_cell/BiasAddBiasAdd"input_layer_lstm/lstm_cell/add:z:09input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
*input_layer_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 input_layer_lstm/lstm_cell/splitSplit3input_layer_lstm/lstm_cell/split/split_dim:output:0+input_layer_lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
"input_layer_lstm/lstm_cell/SigmoidSigmoid)input_layer_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
$input_layer_lstm/lstm_cell/Sigmoid_1Sigmoid)input_layer_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¤
input_layer_lstm/lstm_cell/mulMul(input_layer_lstm/lstm_cell/Sigmoid_1:y:0!input_layer_lstm/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
input_layer_lstm/lstm_cell/ReluRelu)input_layer_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2°
 input_layer_lstm/lstm_cell/mul_1Mul&input_layer_lstm/lstm_cell/Sigmoid:y:0-input_layer_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¥
 input_layer_lstm/lstm_cell/add_1AddV2"input_layer_lstm/lstm_cell/mul:z:0$input_layer_lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
$input_layer_lstm/lstm_cell/Sigmoid_2Sigmoid)input_layer_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
!input_layer_lstm/lstm_cell/Relu_1Relu$input_layer_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2´
 input_layer_lstm/lstm_cell/mul_2Mul(input_layer_lstm/lstm_cell/Sigmoid_2:y:0/input_layer_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
.input_layer_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   o
-input_layer_lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ø
 input_layer_lstm/TensorArrayV2_1TensorListReserve7input_layer_lstm/TensorArrayV2_1/element_shape:output:06input_layer_lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒW
input_layer_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)input_layer_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
#input_layer_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å
input_layer_lstm/whileWhile,input_layer_lstm/while/loop_counter:output:02input_layer_lstm/while/maximum_iterations:output:0input_layer_lstm/time:output:0)input_layer_lstm/TensorArrayV2_1:handle:0input_layer_lstm/zeros:output:0!input_layer_lstm/zeros_1:output:0)input_layer_lstm/strided_slice_1:output:0Hinput_layer_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:09input_layer_lstm_lstm_cell_matmul_readvariableop_resource;input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource:input_layer_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!input_layer_lstm_while_body_18470*-
cond%R#
!input_layer_lstm_while_cond_18469*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
Ainput_layer_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   
3input_layer_lstm/TensorArrayV2Stack/TensorListStackTensorListStackinput_layer_lstm/while:output:3Jinput_layer_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsy
&input_layer_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿr
(input_layer_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(input_layer_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
 input_layer_lstm/strided_slice_3StridedSlice<input_layer_lstm/TensorArrayV2Stack/TensorListStack:tensor:0/input_layer_lstm/strided_slice_3/stack:output:01input_layer_lstm/strided_slice_3/stack_1:output:01input_layer_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskv
!input_layer_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
input_layer_lstm/transpose_1	Transpose<input_layer_lstm/TensorArrayV2Stack/TensorListStack:tensor:0*input_layer_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2l
input_layer_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
)Hidden_layer_Dense1/MatMul/ReadVariableOpReadVariableOp2hidden_layer_dense1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0´
Hidden_layer_Dense1/MatMulMatMul)input_layer_lstm/strided_slice_3:output:01Hidden_layer_Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*Hidden_layer_Dense1/BiasAdd/ReadVariableOpReadVariableOp3hidden_layer_dense1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0²
Hidden_layer_Dense1/BiasAddBiasAdd$Hidden_layer_Dense1/MatMul:product:02Hidden_layer_Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
)Hidden_layer_Dense1/leaky_re_lu/LeakyRelu	LeakyRelu$Hidden_layer_Dense1/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
alpha%>
)Hidden_layer_Dense2/MatMul/ReadVariableOpReadVariableOp2hidden_layer_dense2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0Â
Hidden_layer_Dense2/MatMulMatMul7Hidden_layer_Dense1/leaky_re_lu/LeakyRelu:activations:01Hidden_layer_Dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*Hidden_layer_Dense2/BiasAdd/ReadVariableOpReadVariableOp3hidden_layer_dense2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0²
Hidden_layer_Dense2/BiasAddBiasAdd$Hidden_layer_Dense2/MatMul:product:02Hidden_layer_Dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
(Output_layer_Dense/MatMul/ReadVariableOpReadVariableOp1output_layer_dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0­
Output_layer_Dense/MatMulMatMul$Hidden_layer_Dense2/BiasAdd:output:00Output_layer_Dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)Output_layer_Dense/BiasAdd/ReadVariableOpReadVariableOp2output_layer_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
Output_layer_Dense/BiasAddBiasAdd#Output_layer_Dense/MatMul:product:01Output_layer_Dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity#Output_layer_Dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp+^Hidden_layer_Dense1/BiasAdd/ReadVariableOp*^Hidden_layer_Dense1/MatMul/ReadVariableOp+^Hidden_layer_Dense2/BiasAdd/ReadVariableOp*^Hidden_layer_Dense2/MatMul/ReadVariableOp*^Output_layer_Dense/BiasAdd/ReadVariableOp)^Output_layer_Dense/MatMul/ReadVariableOp2^input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp1^input_layer_lstm/lstm_cell/MatMul/ReadVariableOp3^input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp^input_layer_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2X
*Hidden_layer_Dense1/BiasAdd/ReadVariableOp*Hidden_layer_Dense1/BiasAdd/ReadVariableOp2V
)Hidden_layer_Dense1/MatMul/ReadVariableOp)Hidden_layer_Dense1/MatMul/ReadVariableOp2X
*Hidden_layer_Dense2/BiasAdd/ReadVariableOp*Hidden_layer_Dense2/BiasAdd/ReadVariableOp2V
)Hidden_layer_Dense2/MatMul/ReadVariableOp)Hidden_layer_Dense2/MatMul/ReadVariableOp2V
)Output_layer_Dense/BiasAdd/ReadVariableOp)Output_layer_Dense/BiasAdd/ReadVariableOp2T
(Output_layer_Dense/MatMul/ReadVariableOp(Output_layer_Dense/MatMul/ReadVariableOp2f
1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp1input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp2d
0input_layer_lstm/lstm_cell/MatMul/ReadVariableOp0input_layer_lstm/lstm_cell/MatMul/ReadVariableOp2h
2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp2input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp20
input_layer_lstm/whileinput_layer_lstm/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
Í
!__inference__traced_restore_19755
file_prefix=
+assignvariableop_hidden_layer_dense1_kernel:229
+assignvariableop_1_hidden_layer_dense1_bias:2?
-assignvariableop_2_hidden_layer_dense2_kernel:229
+assignvariableop_3_hidden_layer_dense2_bias:2>
,assignvariableop_4_output_layer_dense_kernel:28
*assignvariableop_5_output_layer_dense_bias:G
4assignvariableop_6_input_layer_lstm_lstm_cell_kernel:	ÈQ
>assignvariableop_7_input_layer_lstm_lstm_cell_recurrent_kernel:	2ÈA
2assignvariableop_8_input_layer_lstm_lstm_cell_bias:	È&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: G
5assignvariableop_16_adam_hidden_layer_dense1_kernel_m:22A
3assignvariableop_17_adam_hidden_layer_dense1_bias_m:2G
5assignvariableop_18_adam_hidden_layer_dense2_kernel_m:22A
3assignvariableop_19_adam_hidden_layer_dense2_bias_m:2F
4assignvariableop_20_adam_output_layer_dense_kernel_m:2@
2assignvariableop_21_adam_output_layer_dense_bias_m:O
<assignvariableop_22_adam_input_layer_lstm_lstm_cell_kernel_m:	ÈY
Fassignvariableop_23_adam_input_layer_lstm_lstm_cell_recurrent_kernel_m:	2ÈI
:assignvariableop_24_adam_input_layer_lstm_lstm_cell_bias_m:	ÈG
5assignvariableop_25_adam_hidden_layer_dense1_kernel_v:22A
3assignvariableop_26_adam_hidden_layer_dense1_bias_v:2G
5assignvariableop_27_adam_hidden_layer_dense2_kernel_v:22A
3assignvariableop_28_adam_hidden_layer_dense2_bias_v:2F
4assignvariableop_29_adam_output_layer_dense_kernel_v:2@
2assignvariableop_30_adam_output_layer_dense_bias_v:O
<assignvariableop_31_adam_input_layer_lstm_lstm_cell_kernel_v:	ÈY
Fassignvariableop_32_adam_input_layer_lstm_lstm_cell_recurrent_kernel_v:	2ÈI
:assignvariableop_33_adam_input_layer_lstm_lstm_cell_bias_v:	È
identity_35¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Â
value¸Bµ#B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ð
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¢
_output_shapes
:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp+assignvariableop_hidden_layer_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_hidden_layer_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp-assignvariableop_2_hidden_layer_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp+assignvariableop_3_hidden_layer_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_output_layer_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_output_layer_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_6AssignVariableOp4assignvariableop_6_input_layer_lstm_lstm_cell_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_7AssignVariableOp>assignvariableop_7_input_layer_lstm_lstm_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_8AssignVariableOp2assignvariableop_8_input_layer_lstm_lstm_cell_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_hidden_layer_dense1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_17AssignVariableOp3assignvariableop_17_adam_hidden_layer_dense1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adam_hidden_layer_dense2_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_hidden_layer_dense2_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_output_layer_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_output_layer_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_22AssignVariableOp<assignvariableop_22_adam_input_layer_lstm_lstm_cell_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_23AssignVariableOpFassignvariableop_23_adam_input_layer_lstm_lstm_cell_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_24AssignVariableOp:assignvariableop_24_adam_input_layer_lstm_lstm_cell_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_hidden_layer_dense1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_hidden_layer_dense1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_hidden_layer_dense2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_hidden_layer_dense2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_output_layer_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_output_layer_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_input_layer_lstm_lstm_cell_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_32AssignVariableOpFassignvariableop_32_adam_input_layer_lstm_lstm_cell_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_33AssignVariableOp:assignvariableop_33_adam_input_layer_lstm_lstm_cell_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 »
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: ¨
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù

D__inference_lstm_cell_layer_call_and_return_conditional_losses_19486

inputs
states_0
states_11
matmul_readvariableop_resource:	È3
 matmul_1_readvariableop_resource:	2È.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/1
Ì8
þ
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17557

inputs"
lstm_cell_17473:	È"
lstm_cell_17475:	2È
lstm_cell_17477:	È
identity¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskâ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_17473lstm_cell_17475lstm_cell_17477*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17472n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_17473lstm_cell_17475lstm_cell_17477*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_17487*
condR
while_cond_17486*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

2__inference_Output_layer_Dense_layer_call_fn_19410

inputs
unknown:2
	unknown_0:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_17961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
°
¾
while_cond_19276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_19276___redundant_placeholder03
/while_while_cond_19276___redundant_placeholder13
/while_while_cond_19276___redundant_placeholder23
/while_while_cond_19276___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
°
¾
while_cond_17679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_17679___redundant_placeholder03
/while_while_cond_17679___redundant_placeholder13
/while_while_cond_17679___redundant_placeholder23
/while_while_cond_17679___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
Ñ	
ÿ
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_17945

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Û7
´
while_body_17825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ÈE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	2È@
1while_lstm_cell_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ÈC
0while_lstm_cell_matmul_1_readvariableop_resource:	2È>
/while_lstm_cell_biasadd_readvariableop_resource:	È¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :æ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ç

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_18986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18986___redundant_placeholder03
/while_while_cond_18986___redundant_placeholder13
/while_while_cond_18986___redundant_placeholder23
/while_while_cond_18986___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
°
¾
while_cond_18841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18841___redundant_placeholder03
/while_while_cond_18841___redundant_placeholder13
/while_while_cond_18841___redundant_placeholder23
/while_while_cond_18841___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
 J

K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19072
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	È=
*lstm_cell_matmul_1_readvariableop_resource:	2È8
)lstm_cell_biasadd_readvariableop_resource:	È
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ô
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_18987*
condR
while_cond_18986*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ù

D__inference_lstm_cell_layer_call_and_return_conditional_losses_19518

inputs
states_0
states_11
matmul_readvariableop_resource:	È3
 matmul_1_readvariableop_resource:	2È.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/1
Ð	
þ
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_19420

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
©#
Ç
while_body_17680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_17704_0:	È*
while_lstm_cell_17706_0:	2È&
while_lstm_cell_17708_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_17704:	È(
while_lstm_cell_17706:	2È$
while_lstm_cell_17708:	È¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0 
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_17704_0while_lstm_cell_17706_0while_lstm_cell_17708_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17620r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_17704while_lstm_cell_17704_0"0
while_lstm_cell_17706while_lstm_cell_17706_0"0
while_lstm_cell_17708while_lstm_cell_17708_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
òK
Ô
!input_layer_lstm_while_body_18634>
:input_layer_lstm_while_input_layer_lstm_while_loop_counterD
@input_layer_lstm_while_input_layer_lstm_while_maximum_iterations&
"input_layer_lstm_while_placeholder(
$input_layer_lstm_while_placeholder_1(
$input_layer_lstm_while_placeholder_2(
$input_layer_lstm_while_placeholder_3=
9input_layer_lstm_while_input_layer_lstm_strided_slice_1_0y
uinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0T
Ainput_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0:	ÈV
Cinput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:	2ÈQ
Binput_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:	È#
input_layer_lstm_while_identity%
!input_layer_lstm_while_identity_1%
!input_layer_lstm_while_identity_2%
!input_layer_lstm_while_identity_3%
!input_layer_lstm_while_identity_4%
!input_layer_lstm_while_identity_5;
7input_layer_lstm_while_input_layer_lstm_strided_slice_1w
sinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensorR
?input_layer_lstm_while_lstm_cell_matmul_readvariableop_resource:	ÈT
Ainput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource:	2ÈO
@input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource:	È¢7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp¢8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp
Hinput_layer_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   û
:input_layer_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0"input_layer_lstm_while_placeholderQinput_layer_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¹
6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAinput_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0ç
'input_layer_lstm/while/lstm_cell/MatMulMatMulAinput_layer_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0>input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ½
8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCinput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0Î
)input_layer_lstm/while/lstm_cell/MatMul_1MatMul$input_layer_lstm_while_placeholder_2@input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
$input_layer_lstm/while/lstm_cell/addAddV21input_layer_lstm/while/lstm_cell/MatMul:product:03input_layer_lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ·
7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBinput_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0Ñ
(input_layer_lstm/while/lstm_cell/BiasAddBiasAdd(input_layer_lstm/while/lstm_cell/add:z:0?input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈr
0input_layer_lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&input_layer_lstm/while/lstm_cell/splitSplit9input_layer_lstm/while/lstm_cell/split/split_dim:output:01input_layer_lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
(input_layer_lstm/while/lstm_cell/SigmoidSigmoid/input_layer_lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*input_layer_lstm/while/lstm_cell/Sigmoid_1Sigmoid/input_layer_lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2³
$input_layer_lstm/while/lstm_cell/mulMul.input_layer_lstm/while/lstm_cell/Sigmoid_1:y:0$input_layer_lstm_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
%input_layer_lstm/while/lstm_cell/ReluRelu/input_layer_lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Â
&input_layer_lstm/while/lstm_cell/mul_1Mul,input_layer_lstm/while/lstm_cell/Sigmoid:y:03input_layer_lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
&input_layer_lstm/while/lstm_cell/add_1AddV2(input_layer_lstm/while/lstm_cell/mul:z:0*input_layer_lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
*input_layer_lstm/while/lstm_cell/Sigmoid_2Sigmoid/input_layer_lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'input_layer_lstm/while/lstm_cell/Relu_1Relu*input_layer_lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Æ
&input_layer_lstm/while/lstm_cell/mul_2Mul.input_layer_lstm/while/lstm_cell/Sigmoid_2:y:05input_layer_lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Ainput_layer_lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ®
;input_layer_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$input_layer_lstm_while_placeholder_1Jinput_layer_lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0*input_layer_lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ^
input_layer_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
input_layer_lstm/while/addAddV2"input_layer_lstm_while_placeholder%input_layer_lstm/while/add/y:output:0*
T0*
_output_shapes
: `
input_layer_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :«
input_layer_lstm/while/add_1AddV2:input_layer_lstm_while_input_layer_lstm_while_loop_counter'input_layer_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
input_layer_lstm/while/IdentityIdentity input_layer_lstm/while/add_1:z:0^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ®
!input_layer_lstm/while/Identity_1Identity@input_layer_lstm_while_input_layer_lstm_while_maximum_iterations^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: 
!input_layer_lstm/while/Identity_2Identityinput_layer_lstm/while/add:z:0^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ¹
!input_layer_lstm/while/Identity_3IdentityKinput_layer_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^input_layer_lstm/while/NoOp*
T0*
_output_shapes
: ©
!input_layer_lstm/while/Identity_4Identity*input_layer_lstm/while/lstm_cell/mul_2:z:0^input_layer_lstm/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2©
!input_layer_lstm/while/Identity_5Identity*input_layer_lstm/while/lstm_cell/add_1:z:0^input_layer_lstm/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
input_layer_lstm/while/NoOpNoOp8^input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp7^input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp9^input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "K
input_layer_lstm_while_identity(input_layer_lstm/while/Identity:output:0"O
!input_layer_lstm_while_identity_1*input_layer_lstm/while/Identity_1:output:0"O
!input_layer_lstm_while_identity_2*input_layer_lstm/while/Identity_2:output:0"O
!input_layer_lstm_while_identity_3*input_layer_lstm/while/Identity_3:output:0"O
!input_layer_lstm_while_identity_4*input_layer_lstm/while/Identity_4:output:0"O
!input_layer_lstm_while_identity_5*input_layer_lstm/while/Identity_5:output:0"t
7input_layer_lstm_while_input_layer_lstm_strided_slice_19input_layer_lstm_while_input_layer_lstm_strided_slice_1_0"
@input_layer_lstm_while_lstm_cell_biasadd_readvariableop_resourceBinput_layer_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
Ainput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resourceCinput_layer_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"
?input_layer_lstm_while_lstm_cell_matmul_readvariableop_resourceAinput_layer_lstm_while_lstm_cell_matmul_readvariableop_resource_0"ì
sinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensoruinput_layer_lstm_while_tensorarrayv2read_tensorlistgetitem_input_layer_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2r
7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp7input_layer_lstm/while/lstm_cell/BiasAdd/ReadVariableOp2p
6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp6input_layer_lstm/while/lstm_cell/MatMul/ReadVariableOp2t
8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp8input_layer_lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_18091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18091___redundant_placeholder03
/while_while_cond_18091___redundant_placeholder13
/while_while_cond_18091___redundant_placeholder23
/while_while_cond_18091___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
ã
ò
)__inference_lstm_cell_layer_call_fn_19437

inputs
states_0
states_1
unknown:	È
	unknown_0:	2È
	unknown_1:	È
identity

identity_1

identity_2¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17472o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/1
ý	
Ü
-__inference_Sequence_LSTM_layer_call_fn_18410

inputs
unknown:	È
	unknown_0:	2È
	unknown_1:	È
	unknown_2:22
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
0__inference_input_layer_lstm_layer_call_fn_18760
inputs_0
unknown:	È
	unknown_0:	2È
	unknown_1:	È
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
­

ì
-__inference_Sequence_LSTM_layer_call_fn_17989
input_layer_lstm_input
unknown:	È
	unknown_0:	2È
	unknown_1:	È
	unknown_2:22
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_17968o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_nameinput_layer_lstm_input
Ö
 
3__inference_Hidden_layer_Dense2_layer_call_fn_19391

inputs
unknown:22
	unknown_0:2
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_17945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
°
¾
while_cond_17486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_17486___redundant_placeholder03
/while_while_cond_17486___redundant_placeholder13
/while_while_cond_17486___redundant_placeholder23
/while_while_cond_17486___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:

½
0__inference_input_layer_lstm_layer_call_fn_18771

inputs
unknown:	È
	unknown_0:	2È
	unknown_1:	È
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¨
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18333
input_layer_lstm_input)
input_layer_lstm_18310:	È)
input_layer_lstm_18312:	2È%
input_layer_lstm_18314:	È+
hidden_layer_dense1_18317:22'
hidden_layer_dense1_18319:2+
hidden_layer_dense2_18322:22'
hidden_layer_dense2_18324:2*
output_layer_dense_18327:2&
output_layer_dense_18329:
identity¢+Hidden_layer_Dense1/StatefulPartitionedCall¢+Hidden_layer_Dense2/StatefulPartitionedCall¢*Output_layer_Dense/StatefulPartitionedCall¢(input_layer_lstm/StatefulPartitionedCall·
(input_layer_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_layer_lstm_inputinput_layer_lstm_18310input_layer_lstm_18312input_layer_lstm_18314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18177Ä
+Hidden_layer_Dense1/StatefulPartitionedCallStatefulPartitionedCall1input_layer_lstm/StatefulPartitionedCall:output:0hidden_layer_dense1_18317hidden_layer_dense1_18319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_17929Ç
+Hidden_layer_Dense2/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense1/StatefulPartitionedCall:output:0hidden_layer_dense2_18322hidden_layer_dense2_18324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_17945Ã
*Output_layer_Dense/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense2/StatefulPartitionedCall:output:0output_layer_dense_18327output_layer_dense_18329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_17961
IdentityIdentity3Output_layer_Dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp,^Hidden_layer_Dense1/StatefulPartitionedCall,^Hidden_layer_Dense2/StatefulPartitionedCall+^Output_layer_Dense/StatefulPartitionedCall)^input_layer_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2Z
+Hidden_layer_Dense1/StatefulPartitionedCall+Hidden_layer_Dense1/StatefulPartitionedCall2Z
+Hidden_layer_Dense2/StatefulPartitionedCall+Hidden_layer_Dense2/StatefulPartitionedCall2X
*Output_layer_Dense/StatefulPartitionedCall*Output_layer_Dense/StatefulPartitionedCall2T
(input_layer_lstm/StatefulPartitionedCall(input_layer_lstm/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_nameinput_layer_lstm_input
ýI

K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_17910

inputs;
(lstm_cell_matmul_readvariableop_resource:	È=
*lstm_cell_matmul_1_readvariableop_resource:	2È8
)lstm_cell_biasadd_readvariableop_resource:	È
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ô
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_17825*
condR
while_cond_17824*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
ò
)__inference_lstm_cell_layer_call_fn_19454

inputs
states_0
states_1
unknown:	È
	unknown_0:	2È
	unknown_1:	È
identity

identity_1

identity_2¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17620o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/1
 J

K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18927
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	È=
*lstm_cell_matmul_1_readvariableop_resource:	2È8
)lstm_cell_biasadd_readvariableop_resource:	È
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ô
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_18842*
condR
while_cond_18841*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

ª
/Sequence_LSTM_input_layer_lstm_while_cond_17300Z
Vsequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_loop_counter`
\sequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_maximum_iterations4
0sequence_lstm_input_layer_lstm_while_placeholder6
2sequence_lstm_input_layer_lstm_while_placeholder_16
2sequence_lstm_input_layer_lstm_while_placeholder_26
2sequence_lstm_input_layer_lstm_while_placeholder_3\
Xsequence_lstm_input_layer_lstm_while_less_sequence_lstm_input_layer_lstm_strided_slice_1q
msequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_cond_17300___redundant_placeholder0q
msequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_cond_17300___redundant_placeholder1q
msequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_cond_17300___redundant_placeholder2q
msequence_lstm_input_layer_lstm_while_sequence_lstm_input_layer_lstm_while_cond_17300___redundant_placeholder31
-sequence_lstm_input_layer_lstm_while_identity
Þ
)Sequence_LSTM/input_layer_lstm/while/LessLess0sequence_lstm_input_layer_lstm_while_placeholderXsequence_lstm_input_layer_lstm_while_less_sequence_lstm_input_layer_lstm_strided_slice_1*
T0*
_output_shapes
: 
-Sequence_LSTM/input_layer_lstm/while/IdentityIdentity-Sequence_LSTM/input_layer_lstm/while/Less:z:0*
T0
*
_output_shapes
: "g
-sequence_lstm_input_layer_lstm_while_identity6Sequence_LSTM/input_layer_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:


!input_layer_lstm_while_cond_18633>
:input_layer_lstm_while_input_layer_lstm_while_loop_counterD
@input_layer_lstm_while_input_layer_lstm_while_maximum_iterations&
"input_layer_lstm_while_placeholder(
$input_layer_lstm_while_placeholder_1(
$input_layer_lstm_while_placeholder_2(
$input_layer_lstm_while_placeholder_3@
<input_layer_lstm_while_less_input_layer_lstm_strided_slice_1U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18633___redundant_placeholder0U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18633___redundant_placeholder1U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18633___redundant_placeholder2U
Qinput_layer_lstm_while_input_layer_lstm_while_cond_18633___redundant_placeholder3#
input_layer_lstm_while_identity
¦
input_layer_lstm/while/LessLess"input_layer_lstm_while_placeholder<input_layer_lstm_while_less_input_layer_lstm_strided_slice_1*
T0*
_output_shapes
: m
input_layer_lstm/while/IdentityIdentityinput_layer_lstm/while/Less:z:0*
T0
*
_output_shapes
: "K
input_layer_lstm_while_identity(input_layer_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
ý	
Ü
-__inference_Sequence_LSTM_layer_call_fn_18387

inputs
unknown:	È
	unknown_0:	2È
	unknown_1:	È
	unknown_2:22
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_17968o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

 __inference__wrapped_model_17405
input_layer_lstm_inputZ
Gsequence_lstm_input_layer_lstm_lstm_cell_matmul_readvariableop_resource:	È\
Isequence_lstm_input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource:	2ÈW
Hsequence_lstm_input_layer_lstm_lstm_cell_biasadd_readvariableop_resource:	ÈR
@sequence_lstm_hidden_layer_dense1_matmul_readvariableop_resource:22O
Asequence_lstm_hidden_layer_dense1_biasadd_readvariableop_resource:2R
@sequence_lstm_hidden_layer_dense2_matmul_readvariableop_resource:22O
Asequence_lstm_hidden_layer_dense2_biasadd_readvariableop_resource:2Q
?sequence_lstm_output_layer_dense_matmul_readvariableop_resource:2N
@sequence_lstm_output_layer_dense_biasadd_readvariableop_resource:
identity¢8Sequence_LSTM/Hidden_layer_Dense1/BiasAdd/ReadVariableOp¢7Sequence_LSTM/Hidden_layer_Dense1/MatMul/ReadVariableOp¢8Sequence_LSTM/Hidden_layer_Dense2/BiasAdd/ReadVariableOp¢7Sequence_LSTM/Hidden_layer_Dense2/MatMul/ReadVariableOp¢7Sequence_LSTM/Output_layer_Dense/BiasAdd/ReadVariableOp¢6Sequence_LSTM/Output_layer_Dense/MatMul/ReadVariableOp¢?Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp¢>Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul/ReadVariableOp¢@Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp¢$Sequence_LSTM/input_layer_lstm/whilej
$Sequence_LSTM/input_layer_lstm/ShapeShapeinput_layer_lstm_input*
T0*
_output_shapes
:|
2Sequence_LSTM/input_layer_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4Sequence_LSTM/input_layer_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4Sequence_LSTM/input_layer_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,Sequence_LSTM/input_layer_lstm/strided_sliceStridedSlice-Sequence_LSTM/input_layer_lstm/Shape:output:0;Sequence_LSTM/input_layer_lstm/strided_slice/stack:output:0=Sequence_LSTM/input_layer_lstm/strided_slice/stack_1:output:0=Sequence_LSTM/input_layer_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-Sequence_LSTM/input_layer_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Ð
+Sequence_LSTM/input_layer_lstm/zeros/packedPack5Sequence_LSTM/input_layer_lstm/strided_slice:output:06Sequence_LSTM/input_layer_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:o
*Sequence_LSTM/input_layer_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    É
$Sequence_LSTM/input_layer_lstm/zerosFill4Sequence_LSTM/input_layer_lstm/zeros/packed:output:03Sequence_LSTM/input_layer_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
/Sequence_LSTM/input_layer_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Ô
-Sequence_LSTM/input_layer_lstm/zeros_1/packedPack5Sequence_LSTM/input_layer_lstm/strided_slice:output:08Sequence_LSTM/input_layer_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:q
,Sequence_LSTM/input_layer_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ï
&Sequence_LSTM/input_layer_lstm/zeros_1Fill6Sequence_LSTM/input_layer_lstm/zeros_1/packed:output:05Sequence_LSTM/input_layer_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
-Sequence_LSTM/input_layer_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
(Sequence_LSTM/input_layer_lstm/transpose	Transposeinput_layer_lstm_input6Sequence_LSTM/input_layer_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&Sequence_LSTM/input_layer_lstm/Shape_1Shape,Sequence_LSTM/input_layer_lstm/transpose:y:0*
T0*
_output_shapes
:~
4Sequence_LSTM/input_layer_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6Sequence_LSTM/input_layer_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6Sequence_LSTM/input_layer_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.Sequence_LSTM/input_layer_lstm/strided_slice_1StridedSlice/Sequence_LSTM/input_layer_lstm/Shape_1:output:0=Sequence_LSTM/input_layer_lstm/strided_slice_1/stack:output:0?Sequence_LSTM/input_layer_lstm/strided_slice_1/stack_1:output:0?Sequence_LSTM/input_layer_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:Sequence_LSTM/input_layer_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,Sequence_LSTM/input_layer_lstm/TensorArrayV2TensorListReserveCSequence_LSTM/input_layer_lstm/TensorArrayV2/element_shape:output:07Sequence_LSTM/input_layer_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¥
TSequence_LSTM/input_layer_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ½
FSequence_LSTM/input_layer_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor,Sequence_LSTM/input_layer_lstm/transpose:y:0]Sequence_LSTM/input_layer_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ~
4Sequence_LSTM/input_layer_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6Sequence_LSTM/input_layer_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6Sequence_LSTM/input_layer_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
.Sequence_LSTM/input_layer_lstm/strided_slice_2StridedSlice,Sequence_LSTM/input_layer_lstm/transpose:y:0=Sequence_LSTM/input_layer_lstm/strided_slice_2/stack:output:0?Sequence_LSTM/input_layer_lstm/strided_slice_2/stack_1:output:0?Sequence_LSTM/input_layer_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÇ
>Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOpGsequence_lstm_input_layer_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	È*
dtype0í
/Sequence_LSTM/input_layer_lstm/lstm_cell/MatMulMatMul7Sequence_LSTM/input_layer_lstm/strided_slice_2:output:0FSequence_LSTM/input_layer_lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈË
@Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIsequence_lstm_input_layer_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0ç
1Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1MatMul-Sequence_LSTM/input_layer_lstm/zeros:output:0HSequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈà
,Sequence_LSTM/input_layer_lstm/lstm_cell/addAddV29Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul:product:0;Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÅ
?Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHsequence_lstm_input_layer_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0é
0Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAddBiasAdd0Sequence_LSTM/input_layer_lstm/lstm_cell/add:z:0GSequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈz
8Sequence_LSTM/input_layer_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.Sequence_LSTM/input_layer_lstm/lstm_cell/splitSplitASequence_LSTM/input_layer_lstm/lstm_cell/split/split_dim:output:09Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split¦
0Sequence_LSTM/input_layer_lstm/lstm_cell/SigmoidSigmoid7Sequence_LSTM/input_layer_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¨
2Sequence_LSTM/input_layer_lstm/lstm_cell/Sigmoid_1Sigmoid7Sequence_LSTM/input_layer_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Î
,Sequence_LSTM/input_layer_lstm/lstm_cell/mulMul6Sequence_LSTM/input_layer_lstm/lstm_cell/Sigmoid_1:y:0/Sequence_LSTM/input_layer_lstm/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
-Sequence_LSTM/input_layer_lstm/lstm_cell/ReluRelu7Sequence_LSTM/input_layer_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ú
.Sequence_LSTM/input_layer_lstm/lstm_cell/mul_1Mul4Sequence_LSTM/input_layer_lstm/lstm_cell/Sigmoid:y:0;Sequence_LSTM/input_layer_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ï
.Sequence_LSTM/input_layer_lstm/lstm_cell/add_1AddV20Sequence_LSTM/input_layer_lstm/lstm_cell/mul:z:02Sequence_LSTM/input_layer_lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¨
2Sequence_LSTM/input_layer_lstm/lstm_cell/Sigmoid_2Sigmoid7Sequence_LSTM/input_layer_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
/Sequence_LSTM/input_layer_lstm/lstm_cell/Relu_1Relu2Sequence_LSTM/input_layer_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Þ
.Sequence_LSTM/input_layer_lstm/lstm_cell/mul_2Mul6Sequence_LSTM/input_layer_lstm/lstm_cell/Sigmoid_2:y:0=Sequence_LSTM/input_layer_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
<Sequence_LSTM/input_layer_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   }
;Sequence_LSTM/input_layer_lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :¢
.Sequence_LSTM/input_layer_lstm/TensorArrayV2_1TensorListReserveESequence_LSTM/input_layer_lstm/TensorArrayV2_1/element_shape:output:0DSequence_LSTM/input_layer_lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
#Sequence_LSTM/input_layer_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 
7Sequence_LSTM/input_layer_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1Sequence_LSTM/input_layer_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©	
$Sequence_LSTM/input_layer_lstm/whileWhile:Sequence_LSTM/input_layer_lstm/while/loop_counter:output:0@Sequence_LSTM/input_layer_lstm/while/maximum_iterations:output:0,Sequence_LSTM/input_layer_lstm/time:output:07Sequence_LSTM/input_layer_lstm/TensorArrayV2_1:handle:0-Sequence_LSTM/input_layer_lstm/zeros:output:0/Sequence_LSTM/input_layer_lstm/zeros_1:output:07Sequence_LSTM/input_layer_lstm/strided_slice_1:output:0VSequence_LSTM/input_layer_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gsequence_lstm_input_layer_lstm_lstm_cell_matmul_readvariableop_resourceIsequence_lstm_input_layer_lstm_lstm_cell_matmul_1_readvariableop_resourceHsequence_lstm_input_layer_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *;
body3R1
/Sequence_LSTM_input_layer_lstm_while_body_17301*;
cond3R1
/Sequence_LSTM_input_layer_lstm_while_cond_17300*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations  
OSequence_LSTM/input_layer_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ³
ASequence_LSTM/input_layer_lstm/TensorArrayV2Stack/TensorListStackTensorListStack-Sequence_LSTM/input_layer_lstm/while:output:3XSequence_LSTM/input_layer_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0*
num_elements
4Sequence_LSTM/input_layer_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
6Sequence_LSTM/input_layer_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6Sequence_LSTM/input_layer_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
.Sequence_LSTM/input_layer_lstm/strided_slice_3StridedSliceJSequence_LSTM/input_layer_lstm/TensorArrayV2Stack/TensorListStack:tensor:0=Sequence_LSTM/input_layer_lstm/strided_slice_3/stack:output:0?Sequence_LSTM/input_layer_lstm/strided_slice_3/stack_1:output:0?Sequence_LSTM/input_layer_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
/Sequence_LSTM/input_layer_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ó
*Sequence_LSTM/input_layer_lstm/transpose_1	TransposeJSequence_LSTM/input_layer_lstm/TensorArrayV2Stack/TensorListStack:tensor:08Sequence_LSTM/input_layer_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
&Sequence_LSTM/input_layer_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¸
7Sequence_LSTM/Hidden_layer_Dense1/MatMul/ReadVariableOpReadVariableOp@sequence_lstm_hidden_layer_dense1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0Þ
(Sequence_LSTM/Hidden_layer_Dense1/MatMulMatMul7Sequence_LSTM/input_layer_lstm/strided_slice_3:output:0?Sequence_LSTM/Hidden_layer_Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¶
8Sequence_LSTM/Hidden_layer_Dense1/BiasAdd/ReadVariableOpReadVariableOpAsequence_lstm_hidden_layer_dense1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ü
)Sequence_LSTM/Hidden_layer_Dense1/BiasAddBiasAdd2Sequence_LSTM/Hidden_layer_Dense1/MatMul:product:0@Sequence_LSTM/Hidden_layer_Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2±
7Sequence_LSTM/Hidden_layer_Dense1/leaky_re_lu/LeakyRelu	LeakyRelu2Sequence_LSTM/Hidden_layer_Dense1/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
alpha%>¸
7Sequence_LSTM/Hidden_layer_Dense2/MatMul/ReadVariableOpReadVariableOp@sequence_lstm_hidden_layer_dense2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0ì
(Sequence_LSTM/Hidden_layer_Dense2/MatMulMatMulESequence_LSTM/Hidden_layer_Dense1/leaky_re_lu/LeakyRelu:activations:0?Sequence_LSTM/Hidden_layer_Dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¶
8Sequence_LSTM/Hidden_layer_Dense2/BiasAdd/ReadVariableOpReadVariableOpAsequence_lstm_hidden_layer_dense2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ü
)Sequence_LSTM/Hidden_layer_Dense2/BiasAddBiasAdd2Sequence_LSTM/Hidden_layer_Dense2/MatMul:product:0@Sequence_LSTM/Hidden_layer_Dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¶
6Sequence_LSTM/Output_layer_Dense/MatMul/ReadVariableOpReadVariableOp?sequence_lstm_output_layer_dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0×
'Sequence_LSTM/Output_layer_Dense/MatMulMatMul2Sequence_LSTM/Hidden_layer_Dense2/BiasAdd:output:0>Sequence_LSTM/Output_layer_Dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
7Sequence_LSTM/Output_layer_Dense/BiasAdd/ReadVariableOpReadVariableOp@sequence_lstm_output_layer_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ù
(Sequence_LSTM/Output_layer_Dense/BiasAddBiasAdd1Sequence_LSTM/Output_layer_Dense/MatMul:product:0?Sequence_LSTM/Output_layer_Dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity1Sequence_LSTM/Output_layer_Dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp9^Sequence_LSTM/Hidden_layer_Dense1/BiasAdd/ReadVariableOp8^Sequence_LSTM/Hidden_layer_Dense1/MatMul/ReadVariableOp9^Sequence_LSTM/Hidden_layer_Dense2/BiasAdd/ReadVariableOp8^Sequence_LSTM/Hidden_layer_Dense2/MatMul/ReadVariableOp8^Sequence_LSTM/Output_layer_Dense/BiasAdd/ReadVariableOp7^Sequence_LSTM/Output_layer_Dense/MatMul/ReadVariableOp@^Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp?^Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul/ReadVariableOpA^Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp%^Sequence_LSTM/input_layer_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2t
8Sequence_LSTM/Hidden_layer_Dense1/BiasAdd/ReadVariableOp8Sequence_LSTM/Hidden_layer_Dense1/BiasAdd/ReadVariableOp2r
7Sequence_LSTM/Hidden_layer_Dense1/MatMul/ReadVariableOp7Sequence_LSTM/Hidden_layer_Dense1/MatMul/ReadVariableOp2t
8Sequence_LSTM/Hidden_layer_Dense2/BiasAdd/ReadVariableOp8Sequence_LSTM/Hidden_layer_Dense2/BiasAdd/ReadVariableOp2r
7Sequence_LSTM/Hidden_layer_Dense2/MatMul/ReadVariableOp7Sequence_LSTM/Hidden_layer_Dense2/MatMul/ReadVariableOp2r
7Sequence_LSTM/Output_layer_Dense/BiasAdd/ReadVariableOp7Sequence_LSTM/Output_layer_Dense/BiasAdd/ReadVariableOp2p
6Sequence_LSTM/Output_layer_Dense/MatMul/ReadVariableOp6Sequence_LSTM/Output_layer_Dense/MatMul/ReadVariableOp2
?Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp?Sequence_LSTM/input_layer_lstm/lstm_cell/BiasAdd/ReadVariableOp2
>Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul/ReadVariableOp>Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul/ReadVariableOp2
@Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp@Sequence_LSTM/input_layer_lstm/lstm_cell/MatMul_1/ReadVariableOp2L
$Sequence_LSTM/input_layer_lstm/while$Sequence_LSTM/input_layer_lstm/while:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_nameinput_layer_lstm_input
°
¾
while_cond_19131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_19131___redundant_placeholder03
/while_while_cond_19131___redundant_placeholder13
/while_while_cond_19131___redundant_placeholder23
/while_while_cond_19131___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
èM

__inference__traced_save_19643
file_prefix9
5savev2_hidden_layer_dense1_kernel_read_readvariableop7
3savev2_hidden_layer_dense1_bias_read_readvariableop9
5savev2_hidden_layer_dense2_kernel_read_readvariableop7
3savev2_hidden_layer_dense2_bias_read_readvariableop8
4savev2_output_layer_dense_kernel_read_readvariableop6
2savev2_output_layer_dense_bias_read_readvariableop@
<savev2_input_layer_lstm_lstm_cell_kernel_read_readvariableopJ
Fsavev2_input_layer_lstm_lstm_cell_recurrent_kernel_read_readvariableop>
:savev2_input_layer_lstm_lstm_cell_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop@
<savev2_adam_hidden_layer_dense1_kernel_m_read_readvariableop>
:savev2_adam_hidden_layer_dense1_bias_m_read_readvariableop@
<savev2_adam_hidden_layer_dense2_kernel_m_read_readvariableop>
:savev2_adam_hidden_layer_dense2_bias_m_read_readvariableop?
;savev2_adam_output_layer_dense_kernel_m_read_readvariableop=
9savev2_adam_output_layer_dense_bias_m_read_readvariableopG
Csavev2_adam_input_layer_lstm_lstm_cell_kernel_m_read_readvariableopQ
Msavev2_adam_input_layer_lstm_lstm_cell_recurrent_kernel_m_read_readvariableopE
Asavev2_adam_input_layer_lstm_lstm_cell_bias_m_read_readvariableop@
<savev2_adam_hidden_layer_dense1_kernel_v_read_readvariableop>
:savev2_adam_hidden_layer_dense1_bias_v_read_readvariableop@
<savev2_adam_hidden_layer_dense2_kernel_v_read_readvariableop>
:savev2_adam_hidden_layer_dense2_bias_v_read_readvariableop?
;savev2_adam_output_layer_dense_kernel_v_read_readvariableop=
9savev2_adam_output_layer_dense_bias_v_read_readvariableopG
Csavev2_adam_input_layer_lstm_lstm_cell_kernel_v_read_readvariableopQ
Msavev2_adam_input_layer_lstm_lstm_cell_recurrent_kernel_v_read_readvariableopE
Asavev2_adam_input_layer_lstm_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Â
value¸Bµ#B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH³
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ì
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_hidden_layer_dense1_kernel_read_readvariableop3savev2_hidden_layer_dense1_bias_read_readvariableop5savev2_hidden_layer_dense2_kernel_read_readvariableop3savev2_hidden_layer_dense2_bias_read_readvariableop4savev2_output_layer_dense_kernel_read_readvariableop2savev2_output_layer_dense_bias_read_readvariableop<savev2_input_layer_lstm_lstm_cell_kernel_read_readvariableopFsavev2_input_layer_lstm_lstm_cell_recurrent_kernel_read_readvariableop:savev2_input_layer_lstm_lstm_cell_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop<savev2_adam_hidden_layer_dense1_kernel_m_read_readvariableop:savev2_adam_hidden_layer_dense1_bias_m_read_readvariableop<savev2_adam_hidden_layer_dense2_kernel_m_read_readvariableop:savev2_adam_hidden_layer_dense2_bias_m_read_readvariableop;savev2_adam_output_layer_dense_kernel_m_read_readvariableop9savev2_adam_output_layer_dense_bias_m_read_readvariableopCsavev2_adam_input_layer_lstm_lstm_cell_kernel_m_read_readvariableopMsavev2_adam_input_layer_lstm_lstm_cell_recurrent_kernel_m_read_readvariableopAsavev2_adam_input_layer_lstm_lstm_cell_bias_m_read_readvariableop<savev2_adam_hidden_layer_dense1_kernel_v_read_readvariableop:savev2_adam_hidden_layer_dense1_bias_v_read_readvariableop<savev2_adam_hidden_layer_dense2_kernel_v_read_readvariableop:savev2_adam_hidden_layer_dense2_bias_v_read_readvariableop;savev2_adam_output_layer_dense_kernel_v_read_readvariableop9savev2_adam_output_layer_dense_bias_v_read_readvariableopCsavev2_adam_input_layer_lstm_lstm_cell_kernel_v_read_readvariableopMsavev2_adam_input_layer_lstm_lstm_cell_recurrent_kernel_v_read_readvariableopAsavev2_adam_input_layer_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesü
ù: :22:2:22:2:2::	È:	2È:È: : : : : : : :22:2:22:2:2::	È:	2È:È:22:2:22:2:2::	È:	2È:È: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	È:%!

_output_shapes
:	2È:!	

_output_shapes	
:È:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	È:%!

_output_shapes
:	2È:!

_output_shapes	
:È:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::% !

_output_shapes
:	È:%!!

_output_shapes
:	2È:!"

_output_shapes	
:È:#

_output_shapes
: 
Ñ

D__inference_lstm_cell_layer_call_and_return_conditional_losses_17472

inputs

states
states_11
matmul_readvariableop_resource:	È3
 matmul_1_readvariableop_resource:	2È.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2È*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates
Ð	
þ
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_17961

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Û7
´
while_body_18092
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ÈE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	2È@
1while_lstm_cell_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ÈC
0while_lstm_cell_matmul_1_readvariableop_resource:	2È>
/while_lstm_cell_biasadd_readvariableop_resource:	È¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :æ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ç

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ó

ÿ
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_19382

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
alpha%>r
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
°
¾
while_cond_17824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_17824___redundant_placeholder03
/while_while_cond_17824___redundant_placeholder13
/while_while_cond_17824___redundant_placeholder23
/while_while_cond_17824___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
­

ì
-__inference_Sequence_LSTM_layer_call_fn_18281
input_layer_lstm_input
unknown:	È
	unknown_0:	2È
	unknown_1:	È
	unknown_2:22
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_nameinput_layer_lstm_input
Û7
´
while_body_19132
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ÈE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	2È@
1while_lstm_cell_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ÈC
0while_lstm_cell_matmul_1_readvariableop_resource:	2È>
/while_lstm_cell_biasadd_readvariableop_resource:	È¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	È*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2È*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :æ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ç

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ó

ÿ
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_17929

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
alpha%>r
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
â

H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18237

inputs)
input_layer_lstm_18214:	È)
input_layer_lstm_18216:	2È%
input_layer_lstm_18218:	È+
hidden_layer_dense1_18221:22'
hidden_layer_dense1_18223:2+
hidden_layer_dense2_18226:22'
hidden_layer_dense2_18228:2*
output_layer_dense_18231:2&
output_layer_dense_18233:
identity¢+Hidden_layer_Dense1/StatefulPartitionedCall¢+Hidden_layer_Dense2/StatefulPartitionedCall¢*Output_layer_Dense/StatefulPartitionedCall¢(input_layer_lstm/StatefulPartitionedCall§
(input_layer_lstm/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_lstm_18214input_layer_lstm_18216input_layer_lstm_18218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18177Ä
+Hidden_layer_Dense1/StatefulPartitionedCallStatefulPartitionedCall1input_layer_lstm/StatefulPartitionedCall:output:0hidden_layer_dense1_18221hidden_layer_dense1_18223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_17929Ç
+Hidden_layer_Dense2/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense1/StatefulPartitionedCall:output:0hidden_layer_dense2_18226hidden_layer_dense2_18228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_17945Ã
*Output_layer_Dense/StatefulPartitionedCallStatefulPartitionedCall4Hidden_layer_Dense2/StatefulPartitionedCall:output:0output_layer_dense_18231output_layer_dense_18233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_17961
IdentityIdentity3Output_layer_Dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp,^Hidden_layer_Dense1/StatefulPartitionedCall,^Hidden_layer_Dense2/StatefulPartitionedCall+^Output_layer_Dense/StatefulPartitionedCall)^input_layer_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2Z
+Hidden_layer_Dense1/StatefulPartitionedCall+Hidden_layer_Dense1/StatefulPartitionedCall2Z
+Hidden_layer_Dense2/StatefulPartitionedCall+Hidden_layer_Dense2/StatefulPartitionedCall2X
*Output_layer_Dense/StatefulPartitionedCall*Output_layer_Dense/StatefulPartitionedCall2T
(input_layer_lstm/StatefulPartitionedCall(input_layer_lstm/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*×
serving_defaultÃ
]
input_layer_lstm_inputC
(serving_default_input_layer_lstm_input:0ÿÿÿÿÿÿÿÿÿF
Output_layer_Dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÛÌ

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
Ë
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

activation

kernel
bias"
_tf_keras_layer
»
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
»
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
_
00
11
22
3
4
&5
'6
.7
/8"
trackable_list_wrapper
_
00
11
22
3
4
&5
'6
.7
/8"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
é
8trace_0
9trace_1
:trace_2
;trace_32þ
-__inference_Sequence_LSTM_layer_call_fn_17989
-__inference_Sequence_LSTM_layer_call_fn_18387
-__inference_Sequence_LSTM_layer_call_fn_18410
-__inference_Sequence_LSTM_layer_call_fn_18281¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z8trace_0z9trace_1z:trace_2z;trace_3
Õ
<trace_0
=trace_1
>trace_2
?trace_32ê
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18574
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18738
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18307
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18333¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z<trace_0z=trace_1z>trace_2z?trace_3
ÚB×
 __inference__wrapped_model_17405input_layer_lstm_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratemm&m'm.m/m0m1m2mvv&v'v.v/v0v1v2v"
	optimizer
,
Eserving_default"
signature_map
5
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Fstates
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32
0__inference_input_layer_lstm_layer_call_fn_18749
0__inference_input_layer_lstm_layer_call_fn_18760
0__inference_input_layer_lstm_layer_call_fn_18771
0__inference_input_layer_lstm_layer_call_fn_18782Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
ö
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18927
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19072
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19217
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19362Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
"
_generic_user_object
ø
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator
[
state_size

0kernel
1recurrent_kernel
2bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
÷
atrace_02Ú
3__inference_Hidden_layer_Dense1_layer_call_fn_19371¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0

btrace_02õ
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_19382¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zbtrace_0
¥
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
,:*222Hidden_layer_Dense1/kernel
&:$22Hidden_layer_Dense1/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
÷
ntrace_02Ú
3__inference_Hidden_layer_Dense2_layer_call_fn_19391¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zntrace_0

otrace_02õ
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_19401¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0
,:*222Hidden_layer_Dense2/kernel
&:$22Hidden_layer_Dense2/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ö
utrace_02Ù
2__inference_Output_layer_Dense_layer_call_fn_19410¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zutrace_0

vtrace_02ô
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_19420¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0
+:)22Output_layer_Dense/kernel
%:#2Output_layer_Dense/bias
4:2	È2!input_layer_lstm/lstm_cell/kernel
>:<	2È2+input_layer_lstm/lstm_cell/recurrent_kernel
.:,È2input_layer_lstm/lstm_cell/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_Sequence_LSTM_layer_call_fn_17989input_layer_lstm_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_Sequence_LSTM_layer_call_fn_18387inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_Sequence_LSTM_layer_call_fn_18410inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
-__inference_Sequence_LSTM_layer_call_fn_18281input_layer_lstm_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18574inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18738inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©B¦
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18307input_layer_lstm_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©B¦
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18333input_layer_lstm_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÙBÖ
#__inference_signature_wrapper_18364input_layer_lstm_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
0__inference_input_layer_lstm_layer_call_fn_18749inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
0__inference_input_layer_lstm_layer_call_fn_18760inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
0__inference_input_layer_lstm_layer_call_fn_18771inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
0__inference_input_layer_lstm_layer_call_fn_18782inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18927inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19072inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±B®
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19217inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±B®
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19362inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Í
}trace_0
~trace_12
)__inference_lstm_cell_layer_call_fn_19437
)__inference_lstm_cell_layer_call_fn_19454½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z}trace_0z~trace_1

trace_0
trace_12Ì
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19486
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19518½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
çBä
3__inference_Hidden_layer_Dense1_layer_call_fn_19371inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_19382inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
çBä
3__inference_Hidden_layer_Dense2_layer_call_fn_19391inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_19401inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
æBã
2__inference_Output_layer_Dense_layer_call_fn_19410inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bþ
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_19420inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_cell_layer_call_fn_19437inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_cell_layer_call_fn_19454inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§B¤
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19486inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§B¤
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19518inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
1:/222!Adam/Hidden_layer_Dense1/kernel/m
+:)22Adam/Hidden_layer_Dense1/bias/m
1:/222!Adam/Hidden_layer_Dense2/kernel/m
+:)22Adam/Hidden_layer_Dense2/bias/m
0:.22 Adam/Output_layer_Dense/kernel/m
*:(2Adam/Output_layer_Dense/bias/m
9:7	È2(Adam/input_layer_lstm/lstm_cell/kernel/m
C:A	2È22Adam/input_layer_lstm/lstm_cell/recurrent_kernel/m
3:1È2&Adam/input_layer_lstm/lstm_cell/bias/m
1:/222!Adam/Hidden_layer_Dense1/kernel/v
+:)22Adam/Hidden_layer_Dense1/bias/v
1:/222!Adam/Hidden_layer_Dense2/kernel/v
+:)22Adam/Hidden_layer_Dense2/bias/v
0:.22 Adam/Output_layer_Dense/kernel/v
*:(2Adam/Output_layer_Dense/bias/v
9:7	È2(Adam/input_layer_lstm/lstm_cell/kernel/v
C:A	2È22Adam/input_layer_lstm/lstm_cell/recurrent_kernel/v
3:1È2&Adam/input_layer_lstm/lstm_cell/bias/v®
N__inference_Hidden_layer_Dense1_layer_call_and_return_conditional_losses_19382\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 
3__inference_Hidden_layer_Dense1_layer_call_fn_19371O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ2®
N__inference_Hidden_layer_Dense2_layer_call_and_return_conditional_losses_19401\&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 
3__inference_Hidden_layer_Dense2_layer_call_fn_19391O&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ2­
M__inference_Output_layer_Dense_layer_call_and_return_conditional_losses_19420\.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_Output_layer_Dense_layer_call_fn_19410O.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿË
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18307	012&'./K¢H
A¢>
41
input_layer_lstm_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18333	012&'./K¢H
A¢>
41
input_layer_lstm_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18574o	012&'./;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
H__inference_Sequence_LSTM_layer_call_and_return_conditional_losses_18738o	012&'./;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
-__inference_Sequence_LSTM_layer_call_fn_17989r	012&'./K¢H
A¢>
41
input_layer_lstm_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
-__inference_Sequence_LSTM_layer_call_fn_18281r	012&'./K¢H
A¢>
41
input_layer_lstm_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_Sequence_LSTM_layer_call_fn_18387b	012&'./;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_Sequence_LSTM_layer_call_fn_18410b	012&'./;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¾
 __inference__wrapped_model_17405	012&'./C¢@
9¢6
41
input_layer_lstm_inputÿÿÿÿÿÿÿÿÿ
ª "GªD
B
Output_layer_Dense,)
Output_layer_DenseÿÿÿÿÿÿÿÿÿÌ
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_18927}012O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 Ì
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19072}012O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ¼
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19217m012?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ¼
K__inference_input_layer_lstm_layer_call_and_return_conditional_losses_19362m012?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ¤
0__inference_input_layer_lstm_layer_call_fn_18749p012O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2¤
0__inference_input_layer_lstm_layer_call_fn_18760p012O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ2
0__inference_input_layer_lstm_layer_call_fn_18771`012?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
0__inference_input_layer_lstm_layer_call_fn_18782`012?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ2Æ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19486ý012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ2
"
states/1ÿÿÿÿÿÿÿÿÿ2
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ2
EB

0/1/0ÿÿÿÿÿÿÿÿÿ2

0/1/1ÿÿÿÿÿÿÿÿÿ2
 Æ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19518ý012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ2
"
states/1ÿÿÿÿÿÿÿÿÿ2
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ2
EB

0/1/0ÿÿÿÿÿÿÿÿÿ2

0/1/1ÿÿÿÿÿÿÿÿÿ2
 
)__inference_lstm_cell_layer_call_fn_19437í012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ2
"
states/1ÿÿÿÿÿÿÿÿÿ2
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ2
A>

1/0ÿÿÿÿÿÿÿÿÿ2

1/1ÿÿÿÿÿÿÿÿÿ2
)__inference_lstm_cell_layer_call_fn_19454í012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ2
"
states/1ÿÿÿÿÿÿÿÿÿ2
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ2
A>

1/0ÿÿÿÿÿÿÿÿÿ2

1/1ÿÿÿÿÿÿÿÿÿ2Û
#__inference_signature_wrapper_18364³	012&'./]¢Z
¢ 
SªP
N
input_layer_lstm_input41
input_layer_lstm_inputÿÿÿÿÿÿÿÿÿ"GªD
B
Output_layer_Dense,)
Output_layer_Denseÿÿÿÿÿÿÿÿÿ