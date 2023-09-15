import os, sys, yaml
import pandas as pd
import numpy as np
import math
from MegaModels.conformal_predictors import ChemFactoryLite
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# used for importing models, making predictions
import joblib
from MegaModels.conformal_predictors import ChemFactoryLite
from MegaModels.models import SKLearnModel


## PARAMETERS
# boolean determining if the script should be executed verbosely
verbose = True
# title of smiles string in curated csv
smiles_column = 'curated_smiles'
# title of dataset value in curated csv
value_column  = 'value'
# default dpi setting
default_dpi = 400

## FUNCTIONS

""" Load smile strings from all files in data dictionary. Col type specifies the column
	information that is pulled from each data sourced. """
def load_data(data_dict, col_type, data_list=None):
	# TODO rename file 

	# inform user ..
	if verbose:
		print("\nLoading SMILES .. ")

	# parse through data
	smiles_array = [""] # array containing smile strings for all data sources
	type_array = [""] # title for each data source

	# if a list of specific data sources was not passed to the method
	if data_dict == None:
		# assume that all of the data should be parsed from
		# the data dictionary
		data_list = data_dict

	for x in data_list:

		# parse the file name and determing if the path exists
		path = data_dict[x]['file_name']
		col = data_dict[x][col_type]
		if not os.path.exists(path):
			print(f"Unable to prase {x}, path ({path}) does not exist.")
			continue # attempt next data source

		# parse smiles from file
		if path.endswith('.tsv'):
			data = pd.read_csv(path, sep = '\t')
		else:
			data = pd.read_csv(path)
		smiles = data[col].tolist()
		data_title = data_dict[x]['data_title']

		# if the array has not been initialized
		if smiles_array[0] == "":
			smiles_array[0] = smiles
			type_array[0] = data_title
		else:
			smiles_array.append(smiles)
			type_array.append(data_title)

		# inform user
		if verbose:
			print(f"Data set {x} contains {len(smiles)} elements.")

	return smiles_array, type_array

""" method that loads data that share similarities """
def load_similar_data(data_dict, op_dict):

	# establish parameters
	simcol = op_dict['similarity'] # col type used to find similarities between data sets
	valcol = op_dict['val_col'] # col type used in value comparison plot
	datax = op_dict['X_comp'] # name of first data set
	datay = op_dict['Y_comp'] # name of second data set
	if verbose:
		print(f"\nLoading DATA .. ")

	df_x = pd.read_csv(data_dict[datax]['file_name']) # data frame for x-axis data
	df_y = pd.read_csv(data_dict[datay]['file_name']) # data frame for y-axis data
	if verbose:
		print(f"X DATA ({datax}) in FILE ({data_dict[datax]['data_title']}) has POINTS ({len(df_x.index)})")
		print(f"Y DATA ({datay}) in FILE ({data_dict[datay]['data_title']}) has POINTS ({len(df_y.index)})")

	# find enteries that share the same value in the 'similarity' column
	x_data = [] # empty data set that contains similar enteries
	y_data = []
	# column titles for finding similarities
	x_simcol = data_dict[datax][simcol] 
	y_simcol = data_dict[datax][simcol]
	# column titles for parsing comparison values
	x_valcol = data_dict[datax][valcol]
	y_valcol = data_dict[datax][valcol]
	if verbose:
		print(f"Finding enteries in datasets that share {simcol} values ..")
	for i in range(0,len(df_x.index)):
		xsim = df_x.at[i,x_simcol]
		ysim = df_y.loc[df_y[y_simcol] == xsim]
		if len(ysim.index) == 1:
			# print(f"{i} :: X VAL ({xval}) :: Y VAL ({yval[0]})")
			xval = df_x.at[i, x_valcol]
			yval = ysim[y_valcol].tolist()
			x_data.append(xval)
			y_data.append(yval[0])
		elif len(ysim.index) > 1:
			print(f"ERROR :: Y data set contains multiple enteries of {xval}.")

	if verbose:
		print(f"{len(x_data)} enteries found in datasets that have the same {simcol} values.")

	return x_data, y_data

# method for translating all smiles to fingerprints
def smiles_2_fp (smiles_array, fp_type):
	# inform user ..
	if verbose:
		print(f"\nTranslating SMILES to {fp_type} fingerprint type.")

	# loop through smile string, translate to fp
	fp_array = [""] * len(smiles_array)
	for x in range(0, len(smiles_array)):
		fp = [""] * len(smiles_array[x])
		chem_fact = ChemFactoryLite()
		chem_fact.add_smiles(smiles_array[x]) # add smiles to chem factory object
		fp = chem_fact.smiles_2_fp(fp_type=f'{fp_type}') # use chem factory to convert 
		# smiles to fingerprints and return
		fp_array[x] = fp

		# report to user
		if verbose:
			print(f"{len(fp)} smile strings converted to type {fp_type} fingerprints.")

	return fp_array

# method that combines smiles from different sources into one array
def merge(fp_array, data_type):
	# report to user
	if verbose:
		print(f"\nMerging data ..")

	# initialize arrays
	i = 0
	x_data = [""] * len(fp_array[i])
	x_type = [""] * len(fp_array[i])

	# add the first data set to the array
	for x in range(0, len(fp_array[i])):
		x_data[x] = fp_array[i][x]
		x_type[x] = data_type[i]

	# if there are more than one data set
	if len(data_type) > 1:
		# loop through the rest of the data sets
		for i in range(1, len(fp_array)):
			for x in range(0,len(fp_array[i])):
				# append additional data to array
				x_data.append(fp_array[i][x])
				x_type.append(data_type[i])

	if verbose:
		print(f"Data from {i+1} data sources merged to {len(x_data)} total data points.")

	return x_data, x_type

def perform(op, op_dict, data_dict, set_dict, model_dict):

	# TODO pass settings dict to method
	# OR return figure to method
	if op_dict['type'] == "val_dist":
		plot_value_distribution(op_dict, data_dict, set_dict)
	elif op_dict['type'] == "val_comparison":
		plot_comparison(set_dict, op_dict, data_dict)
	elif op_dict['type'] == "chem_diversity":
		plot_chem_diversity(op_dict, data_dict)
	elif op_dict['type'] == 'binary_distribution':
		plot_binary_value_distribution(op_dict, data_dict, set_dict)
	elif op_dict['type'] == 'binary_model_true_predict':
		plot_binary_model_true_predictions(op_dict, model_dict, set_dict)
	elif op_dict['type'] == 'regression_model_true_predict':
		plot_regression_model_true_predictions(op_dict, model_dict, set_dict)
	elif op_dict['type'] == 'make_predictions':
		make_predictions(op_dict, model_dict, data_dict, set_dict)
	elif op_dict['type'] == 'classification_distribution':
		plot_classification_distribution(op_dict, data_dict, set_dict)
	elif op_dict['type'] == 'model_stat_comparison':
		plot_model_stat_comparison(op_dict, model_dict, set_dict)
	elif op_dict['type'] == 'truth_table':
		plot_truth_table(op_dict, data_dict, model_dict, set_dict)
	# elif op_dict['type'] == 'model_predictions_regression'
	else:
		print(f"TODO :: Implement OPERATION ({op_dict['type']})")
		exit()

def plot_chem_diversity(op_dict, data_dict):

	# load smiles from 
	# load data
	x_smiles, data_title = load_data(data_dict, op_dict['val_col'])

	# convert smiles to fingerprints
	x_fp = smiles_2_fp(x_smiles,op_dict['fp_type'])

	# combine strings, type into one array
	x_data, x_type = merge(x_fp, data_title)

	plot_type = op_dict['plot_type']
	if verbose:
		print(f"\nGenerating {plot_type} CHEM DIVERSITY plot.")

	if plot_type == 'PCA':

		# perform PCA
		if verbose:
			print(f"Performing PCA analysis .. ")
		pca = PCA()
		projection = pca.fit_transform(x_data)

		# calculate the error associated with each PCA component
		if verbose:
			print(f"Calculating PCA error for N ({len(pca.singular_values_)}) dimensions ..")
		PCA_dims = [0] * pca.n_components_
		sing_vals = [0] * pca.n_components_ 
		pca_err = [0.] * pca.n_components_
		tot_err = 0.
		for i in range(pca.n_components_):
			PCA_dims[i] = (i+1) # list of PCA dimenions as integers
			sing_vals[i] = pca.singular_values_[i] # list of singular values
			tot_err += sing_vals[i] ** 2 # total error is accumulation of each singular value square
			pca_err[i] = math.sqrt(tot_err) # error associated with recreation of data up to that many dimensions
		tot_err = math.sqrt(tot_err)
		for i in range(pca.n_components_):
			# normalize error for each pca dimension by the total error
			pca_err[i] = (pca_err[i] / tot_err) * 100
	

		# create dataframes for each plot
		df_PCAplot = pd.DataFrame(dict(PCA_1=projection[:,0], PCA_2=projection[:,1], dataset=x_type))
		df_SingVals = pd.DataFrame(dict(dims=PCA_dims[:], SV=sing_vals[:]))
		df_PCAerr = pd.DataFrame(dict(dims=PCA_dims[:], ERR=pca_err[:]))
		if verbose:
			print(f"Generating graph ..") 

		# generate first plot :: first two PCA dimensions
		palette = sns.color_palette("bright")
		fig1 = sns.scatterplot(data=df_PCAplot, x='PCA_1', y='PCA_2', hue='dataset'\
			, palette=palette, s=7)
		fig1.set(title="First Two PCA Dimensions for Dataset (Error = {:.1f}%)".format(pca_err[1]),\
			xlabel="$PCA_{{1}}$", ylabel="$PCA_{{2}}$")
		# fig.set_box_aspect(1)
		path = op_dict['save_as'] + '_PCAanal.tiff'
		plt.savefig(path, dpi = 600, bbox_inches = 'tight')

		# second subplot is histogram of singular values from PCA Analysis
		# plot the singular vales
		plt.clf()
		fig = sns.lineplot(data=df_SingVals, x = 'dims', y = 'SV', color = 'blue')
		fig.fill_between(df_SingVals.dims.values, df_SingVals.SV.values, alpha=0.5, color = 'blue')
		fig.set(title="Singular Values for PCA",\
			xlabel="PCA Dimensions", ylabel = r"Log Plot of Singular Values ($\sigma$)", yscale='log')
		ax1_patch = mpatches.Patch(color = 'blue', label = 'Singular Values')
		fig.legend(handles=[ax1_patch], loc="upper left")
		# axs[1].set_box_aspect(1)
		# plot the error against the singular values
		twin_ax = fig.twinx()
		sns.lineplot(ax=twin_ax, data = df_PCAerr, x = 'dims', y = 'ERR', color = 'red')
		twin_ax.set(ylabel="PCA Error (%)")
		twin_ax.fill_between(df_SingVals.dims.values, df_PCAerr.ERR.values, alpha=0.5, color = 'red')
		twin_ax_patch = mpatches.Patch(color = 'red', label = 'Error')
		twin_ax.legend(handles=[twin_ax_patch], loc = "upper right")
		path = op_dict['save_as'] + "_PCAerror.tiff"
		plt.savefig(path, dpi = 600, bbox_inches = 'tight')

	# elif anal_type == 'tSNE':

	# 	## DATA SET TOO LARGE, USE UMAP

	# 	# change x_data to np array
	# 	X_PCA = np.array(x_data, dtype=float)

	# 	# reduce dimensionality of data set to 512
	# 	pca = PCA(n_components=50)
	# 	x_TSNE = pca.fit_transform(X_PCA)

	# 	# perform tSNE
	# 	tsne = TSNE(n_components=2)
	# 	projection = tsne.fit_transform(x_TSNE)
	# 	print("tSNE analysis completed.")

	# 	# plot
	# 	fig = plt.scatter(projection[:,0], projection[:,1], s = 2, c = y_data, vmin = min(y_data), vmax=max(y_data), cmap = 'summer')
	# 	plt.colorbar(fig, label = "$-log(Molarity)$")
	# 	plt.xlabel("tSNE_1")
	# 	plt.ylabel("tSNE_2")
	# 	plt.suptitle("tSNE Projection of Data Set (n = 2)")
	# 	plt.title("(targret = {:s}; finger print = {:s})".format(target,fp_type))
	# 	plt.show()

def plot_comparison(set_dict, op_dict, data_dict):

	# load settings
	title = f"Comparison of Values for {op_dict['X_comp']} and {op_dict['Y_comp']} Datasets"
	if 'title' in op_dict:
		title = op_dict['title']

	dpi_set = 400
	if 'dpi' in set_dict:
		dpi_set = int(set_dict['dpi'])

	palette_set = 'rocket'
	if 'palette' in set_dict:
		palette_set = set_dict['palette']

	x_axis_label = f"{op_dict['X_comp']} Values"
	if 'x_axis_label' in op_dict:
		x_axis_label = op_dict['x_axis_label']

	y_axis_label = f"{op_dict['Y_comp']} Values"
	if 'y_axis_label' in op_dict:
		y_axis_label = op_dict['y_axis_label']

	# load similar data
	x_data, y_data = load_similar_data(data_dict, op_dict)

	# plot similarity data
	if verbose:
		print(f"\nPLOTTING DATA .. ")

	# apply linear regression
	p = np.poly1d(np.polyfit(x_data, y_data, 1))
	x_line = np.linspace(np.amin(x_data), np.amax(x_data), 200)
	fig, ax = plt.subplots()
	sns.regplot(x = x_data, y = y_data, scatter_kws={'s':2}, \
		line_kws={'label': 'Linear regression line','color': 'm'}, label="Dataset")
	ax.plot([], [], ' ', label="{0}".format(p))
	ax.legend(loc="upper left")

	plt.title(title)
	ax.set_xlabel(x_axis_label)
	ax.set_ylabel(y_axis_label)
	path = op_dict['save_to'] + op_dict['X_comp'] + "_" + op_dict['Y_comp'] + "IC50_comparison.tiff"
	plt.savefig(path, dpi = 600, bbox_inches = 'tight')
	plt.show()

def plot_value_distribution(op_dict, data_dict, set_dict):

	# load settings
	dpi_set = int(set_dict['dpi'])
	palette_set = set_dict['palette']
	title = op_dict['title']
	subtitle = op_dict['subtitle']
	
	# load, merge data
	x_data, data_title = load_data(data_dict, op_dict['val_col'], op_dict['data'])
	x_data, x_type = merge(x_data, data_title)

	# combine set into data frame and plot
	data = pd.DataFrame(dict(Value=x_data, Dataset=x_type))
	sns.displot(data, x = 'Value', hue = 'Dataset', palette = palette_set, element='step' )
	plt.suptitle(subtitle)
	plt.title(title)
	path = op_dict['save_as'] + op_dict['data'][0] + "_distribution.tiff"
	plt.savefig(path, dpi = dpi_set, bbox_inches = 'tight')

def plot_classification_distribution (op_dict, data_dict, set_dict):

	if verbose:
		print(f"\nPlotting classification distribution for: {op_dict['data']}")

	# load settings
	dpi_set = set_dict['dpi']
	palette_set = set_dict['palette']
	title_set = op_dict['title']

	# loaded dataset(s), merge
	if verbose:
		print("Loading data ..")
	x_data, x_type = load_data(data_dict, op_dict['class_col'])
	x_data, x_type = merge(x_data, x_type)

	# determine class distribution
	if verbose:
		print(f"Classifying {len(x_data)} data points .. ")
	labels = op_dict['class']
	c = [""] * len(x_data) 
	# initialize as empty label
	for i in range(len(x_data)):
		# loop through each data point
		# determine if class if present
		has_label = False
		for l in labels:
			if l.lower() in x_data[i].lower():
				c[i] = l
				has_label = True 
				break

		if has_label == False:
			c[i] = "Other / \nNo Label"

	# create dataframe, sort by count
	dist = pd.DataFrame(dict(Strain=c))
	dist_sort = pd.DataFrame(dict(Strain=c))
	order = dist['Strain'].value_counts()
	i = 0
	for idx in dist['Strain'].value_counts().index:
		for j in range(order[idx]):
			dist_sort.at[i, 'Strain'] = idx
			i += 1

	# plot sorted distribution
	g = sns.displot(data=dist_sort, x="Strain", discrete=True)
	plt.title(title_set)
	plt.suptitle(f"(N = {i})")
	plt.xlabel("Plasmodium f. Strain")
	plt.ylabel("Count")
	# plt.show()

	# save figure
	plt.gcf().set_size_inches(10,7)
	path = op_dict['save_to'] + "ChEMBL_strain_distribution.tiff"
	plt.savefig(path, dpi = dpi_set, bbox_inches = 'tight')

""" method for plotting data distrubution of data
	used for binarization. 
	
	Binary data is slightly different
	than a distribution of binary values, because it accounts
	relationships that qualify an activity as greater than
	or less than a certain value, in addition to being equal to"""
def plot_binary_value_distribution(op_dict, data_dict, set_dict):

	if verbose:
		print(f"\nPlotting binary distribution of values in DATASET ({op_dict['data']}).")


	# load settings
	title = f"Distribution of Values Sorted by Their Standard Relationship"
	if 'title' in op_dict:
		title = op_dict['title']

	dpi_set = 400
	if 'dpi' in set_dict:
		dpi_set = int(set_dict['dpi'])

	palette_set = 'rocket'
	if 'palette' in set_dict:
		palette_set = set_dict['palette']

	# load values, merge
	x_val_data, x_val_source = load_data(data_dict, op_dict['val_col'], op_dict['data'])
	x_val_data, x_val_source = merge(x_val_data, x_val_source)

	# load relationships, merge
	x_relation, x_relation_source = load_data(data_dict, op_dict['relation'], op_dict['data'])
	x_relation, x_relation_source = merge(x_relation, x_relation_source)

	# create data frame merging the values and the relationships
	data = pd.DataFrame(dict(Value=x_val_data, Relationship=x_relation))

	# get information for labeling 
	num = len(x_relation)

	# get unique relationships
	r = data['Relationship'].unique().tolist()
	# count the number of entries with each relationship
	n_r = [i for i in range(len(r))] # count for each relation
	for i in range(len(r)):
		n_r[i] = len(data.loc[data['Relationship'] == r[i]])
	# add the count to the label for each relationship
	# create dictionary of replacement values
	replace = {}
	for i in range(len(r)):
		# for entries that contain eq, lt, gt relationships 
		# TODO :: created mapping function for mapping array
		new_label = ""
		if r[i] == 'eq':
			new_label = "Equal"
		elif r[i] == 'lt':
			new_label = "Less Than"
		elif r[i] == 'gt':
			new_label = "Greater Than"
		else:
			new_label = r[i]
		new_label = new_label + " (N = {:})".format(n_r[i])
		replace[r[i]] = new_label
	data['Relationship'] = data['Relationship'].map(replace)
	r = data['Relationship'].unique().tolist()

	# plot data in seaborn distribution
	sns.displot(data, x = 'Value', hue = 'Relationship', palette = set_dict['palette'], element = 'step') #, multiple='stack'
	plt.suptitle(f"N = {num}")
	plt.title(title)
	# plt.legend(loc='upper right')

	# save to path
	data_list = ""
	for d in op_dict['data']:
		data_list += d + "_"
	path = op_dict['save_to'] + data_list + "binarydist.tiff"
	plt.savefig(path, dpi = 200, bbox_inches = 'tight')
	plt.show()


""" method for plotting the true and false predictions of a binary
	model generated by MegaModel building"""
def plot_binary_model_true_predictions (op_dict, model_dict, set_dict):

	if verbose:
		print(f"\nPlotting binary distribution of predictions in MODEL ({op_dict['model']}).")

	# load settings
	dpi_set = 400
	if 'dpi' in set_dict:
		dpi_set = int(set_dict['dpi'])

	palette_set = 'rocket'
	if 'palette' in set_dict:
		palette_set = set_dict['palette']

	# load model details
	data = op_dict['data']
	model = op_dict['model']
	fp_type = model_dict[model]['fp_type']
	model_type = model_dict[model]['model_type']
	model_dir = model_dict[model]['model_dir']

	# load settings
	title = f"Binary Active and Inactive Predictions from {model}"
	if 'title' in op_dict:
		title = op_dict['title']

	subtitle = f"(dataset = {data}, fp_type = {fp_type}, model_type = {model_type})"
	if 'subtitle' in op_dict:
		subtitle = op_dict['subtitle']

	# Set up seaborn basics
	sns.set_theme(style='white', palette=palette_set) # , font_scale=1.5, context='poster', 

	#sns.set_context('poster')  # paper, notebook, talk, poster
	#sns.set_context('poster', font_scale=1., rc={'grid.linewidth': 1.})  # Setting a few options
	#sns.plotting_context()  # Returns rc dictionary (so you can see parameters)

	#sns.set_style(style=None, rc=None)  # darkgrid, whitegrid, dark, white, ticks
	#sns.axes_style()  # Returns axis style parameters

	dataset_path = model_dir + 'true_pred_' + model_dict[model]['model_type'] + \
		'_' + model_dict[model]['fp_type'] + '.csv'
	stats_path = model_dict[model]['model_dir'] + 'stats_' + model_dict[model]['model_type'] + \
		'_' + model_dict[model]['fp_type'] + '.csv'
	data_x_column =   'y_prob'
	category_column = 'y_true'

	print(f"Loading {model} from {dataset_path}.")

	# load data
	df = pd.read_csv(dataset_path) # data set containing predictions
	stats = pd.read_csv(stats_path)
	auc = stats['auc'].tolist()

	# plot data
	fig, ax = plt.subplots() # figsize=(25,25)
	ax.set_xlabel('Probability-Like Score')
	ax.set_ylabel('Number of Compounds')
	plt.suptitle(title)
	ax.set_title(subtitle)

	g = sns.histplot(data=df, x=data_x_column, hue=category_column, stat='count', kde=True, ax=ax,
	                 hue_order=[1, 0], binwidth=0.025, binrange=[0.,1.001], legend=False)
	ax.plot([], [], ' ')


	#g = sns.jointplot(data=df, x=data_x_column, y=data_y_column, kind='reg', truncate=False,
	#                  xlim=(4,10), ylim=(4,10), color='b', height=20, 
	#                  fit_reg=False, ci=None)
	#g.set_axis_labels('True Activity [-log(M)]', 'Predicted Activity [-log(M)]')

	# plt.ylim(0,35)
	plt.legend(loc='upper right', labels=['Inactive', 'Active', "AUC = {:.3f}".format(auc[0])]) #, fontsize=30

	# display plot, or save
	# plt.show()
	save_path = ""
	save_path = op_dict['save_to']	
	if op_dict['save_to'] == 'model_dir':
		save_path = model_dict[model]['model_dir']
	save_path += op_dict['model'] + '_binary_predictions.tiff'
	plt.savefig(save_path, dpi = dpi_set)

""" method for plotting the binary model performance statistics of 
	several different binary models against one another."""
def plot_model_stat_comparison(op_dict, model_dict, set_dict):

	if verbose:
		print(f"\nComparing model stats ..")

	# load figure settings
	dpi_set = 400
	if 'dpi' in set_dict:
		dpi_set = int(set_dict['dpi'])

	palette_set = 'rocket'
	if 'palette' in set_dict:
		palette_set = set_dict['palette']

	# create an empty dataframe that contains the stats and model
	df = pd.DataFrame(0., columns = op_dict['stats'], index = op_dict['models'])

	# loop through models, get stats and add to df
	for m in op_dict['models']:
		# for each model
		# get stats from file
		stats_path = model_dict[m]['model_dir'] + 'stats_' + model_dict[m]['model_type'] + '_' + \
			model_dict[m]['fp_type'] + '.csv'
		m_stat = pd.read_csv(stats_path)

		for s in op_dict['stats']:
			# get stats, add to df
			df.at[m, s] = m_stat.at[0, s].astype(float)


	# get graph labels, print
	title = f"Comparison of {op_dict['model_type']} Model Performance Statistics"
	if title in op_dict:
		title = op_dict['title']

	subtitle = f"(Dataset = {op_dict['data']})"
	if subtitle in op_dict:
		subtitle = op_dict['subtitle']

	sns.heatmap(df, annot=True, linewidths=2, linecolor='black', fmt = '.2f')
	plt.suptitle(title)
	plt.title(subtitle)
	plt.xlabel(f"{op_dict['model_type']} Model Scores")
	plt.ylabel(f"{op_dict['model_type']} Models")

	# display plot, or save
	# plt.show()
	save_path = ""
	save_path = op_dict['save_to']	
	if op_dict['save_to'] == 'model_dir':
		save_path = model_dict[model]['model_dir']
	save_path += op_dict['data'] + '_binary_model_comparison.tiff'
	plt.savefig(save_path, dpi = dpi_set, bbox_inches = 'tight')
	plt.show()


""" method for plotting the predictions made for a regression model test
	set (5-fold cross validation) against the true value. """
def plot_regression_model_true_predictions (op_dict, model_dict, set_dict):


	# establish parameters
	model = op_dict['model']
	model_type = model_dict[model]['model_type']
	fp_type = model_dict[model]['fp_type']
	model_name = model_type + "_" + fp_type
	model_dir_path = model_dict[model]['model_dir']
	if verbose:
		print(f"\nLoading MODEL {model} from DIR {model_dir_path} ... ")

	# load settings
	title = op_dict['title']
	subtitle = op_dict['subtitle']
	dpi_set = int(set_dict['dpi'])
	palette_set = set_dict['palette']

	# get model stats
	stats_file_name = "stats_" + model_name + ".csv"
	stats_path = model_dir_path + stats_file_name
	if verbose:
		print("Pulling model stats from " + stats_path)
	stats = pd.read_csv(stats_path)
	rmse = stats['rmse'].tolist()
	r2 = stats['r2'].tolist()

	# pull model predictions
	predict_file_name = "true_pred_" + model_name + ".csv"
	predict_path = model_dir_path + predict_file_name
	if verbose:
		print("Pulling model predictions from " + predict_path)
	predict = pd.read_csv(predict_path)
	y_data = predict['y_true'].tolist()
	y_test = predict['y_pred'].tolist()

	# fit line to known values and predictions
	p = np.poly1d(np.polyfit(y_data, y_test, 1))
	x_line = np.linspace(np.amin(y_data), np.amax(y_data), 200)

	# plot scatter and regression line
	fig, ax = plt.subplots()
	# sns.jointplot(x=y_data, y=y_test, kind='reg', truncate=False,
	# #                  xlim=(4,10), ylim=(4,10), 
	#                   color='b', height=20, 
	#                   fit_reg=False, ci=None)
	ax.plot([0.5,11.5], [0.5,11.5], color='black', linestyle='--', label = "y = x")
	g = sns.regplot(x = y_data, y = y_test, scatter_kws={'s':2}, line_kws={'label': 'Linear regression line','color': 'm'}, label="Original data")
	g.set_xlim(0, 12)
	g.set_ylim(0, 12)

	# add additional information to legend, add legend to subplot
	ax.plot([], [], ' ', label="{0}".format(p))
	# ax.plot([], [], ' ', label="RMSE = {:.3f}".format(rmse[0]))
	# ax.plot([], [], ' ', label="R2 = {:.3f}".format(r2[0]))
	ax.legend(loc="upper left")

	# add labels
	ax.set_xlabel("Known Value ($-log(Molarity)$)", fontsize = 12)
	ax.set_ylabel("Predicted Value ($-log(Molarity)$)", fontsize = 12)
	plt.suptitle(title, fontsize = 12)
	plt.title(subtitle, fontsize = 10)

	# save plot
	# plt.show()
	save_path = "" 
	save_name = model + "_regression_predictions.tiff"
	if op_dict['save_to'] == 'model_dir':
		save_path = model_dir_path
	else:
		save_path = op_dict['save_to']

	plt.savefig(save_path + save_name, dpi = dpi_set)

""" method for comparing the true values and predictions made
	by a model for a dataset that has experimentally obtained values 

	NOTE :: this program assumes that values are in negative log(M)
	"""
def make_predictions (op_dict, model_dict, data_dict, set_dict):

	# inform user
	if verbose:
		print(f"\nPlotting Predictions of DATASET ({op_dict['data']}) with MODEL ({op_dict['model']})")

	# load settings
	dpi_set = default_dpi
	if 'dpi' in set_dict:
		dpi_set = int(set_dict['dpi'])

	title = f"Predicted Activity of {op_dict['data']} vs. Measured Activity"
	if 'title' in op_dict['title']:
		title = op_dict['title']
	
	subtitle = f"(model = {op_dict['model']})"
	if 'subtitle' in op_dict:
		subtitle = op_dict['subtitle']

	x_axis_label = "Probability-Like Score from Model"
	if 'x_axis_label' in op_dict:
		x_axis_label = op_dict['x_axis_label']

	y_axis_label = "Measured Activity"
	if 'y_axis_label' in op_dict:
		y_axis_label = op_dict['y_axis_label']

	# load model
	model_name = op_dict['model'] # model name
	model_config = model_dict[model_name] # model config file
	model_path = model_config['model_dir'] + model_config['model_name']
	if verbose:
		print ("Loading model in: " + model_path)
	fp_type = model_config['fp_type'] # finger print used to create QSAR relationship
	model = joblib.load(model_path) # model generated by MegaModels package

	# get dataset
	data_name = op_dict['data'] # data set name
	data_config = data_dict[data_name] # data set config file
	data_path = data_config['data_dir'] + data_config['data_file']
	if verbose:
		print ("Loading data in: " + data_path)
	smiles_column = data_config['smile_column'] # column containing similes strings
	value_column = data_config['value_column'] # column containing values

	# get smile strings from data set, convert to fingerprints
	if verbose:
		print(f"Converting SMILE strings to {fp_type} ..")
	chem_fact = ChemFactoryLite() # object used to generate fingerprints from strings
	chem_fact.load_data(data_path, smiles_col=smiles_column, val_col=value_column)
	fp_data, y_data = chem_fact.get_data(Smiles=False, fp_type=fp_type)

	# if the data set has standard deviation values, pull them from the file
	y_err = []
	has_range = False
	if ('range_upper_column' in data_config) and ('range_lower_column' in data_config):
		has_range = True
		if verbose:
			print("Loading value range from: " + data_path)

		# create data frame containing std values
		data_df = pd.read_csv(data_path)
		y_low_range = data_df[data_config['range_lower_column']].tolist()
		y_up_range = data_df[data_config['range_upper_column']].tolist()

		# for the lower range values
		y_low_err = []
		for i in range(len(y_low_range)):
			# the lower bounds is the difference between the calculated value
			# and the lower bounds
			y_low_err.append(y_data[i] - y_low_range[i])

		# for the upper range values
		y_up_err = []
		for i in range(len(y_up_range)):
			# the upper bounds is the difference between the upper bounds and 
			# the calculated value
			y_up_err.append(y_up_range[i] - y_data[i])

		# calculate the range as the difference between the upper and lower values
		y_err = [y_low_err, y_up_err]

	# make predictions
	# NOTE :: this method assumes that the model is binary (ie not regression)
	# returns the probabilities that compound is either: [0] inactive, [1] active
	if verbose:
		print(f"Predicting data set values ..")
	if model_dict[op_dict['model']]['model_classification'] == 'binary':
		# for binary models
		probs = model.predict_proba(fp_data)
	elif model_dict[op_dict['model']]['model_classification'] == 'regression':
		# for regression models
		probs = model.predict(fp_data)
	else:
		print(f"Predictions for {model_dict[op_dict['model']]['model_classification']} model type not implemented yet.")
		exit()

	# get the probability-like score of the compounds activity
	x_prob = []
	if model_dict[op_dict['model']]['model_classification'] == "binary":
		for k in range(len(probs)):
			x_prob.append(probs[k][1])
	elif model_dict[op_dict['model']]['model_classification'] == "regression":
		x_prob = probs

	# plot the probability-like predictions of activity against
	# the real, experimentally measured activity of the compounds
	# df = pd.DataFrame(dict(Probabilty=x_prob, Value=))

	if has_range:
		plt.errorbar(x_prob, y_data, yerr=y_err, marker='o',markersize=3, color='black', linestyle='none', capsize=3, label = "Data Points")
		plt.ylim(min(min(y_low_range) - 0.1, 0), max(max(y_up_range) + 0.1, 12))
		if model_dict[op_dict['model']]['model_classification'] == "binary":
			plt.xlim(0,1)
		elif model_dict[op_dict['model']]['model_classification'] == "regression":
			plt.xlim(0,12)
	else:
		plt.scatter(x_prob, y_data, marker='o', s=3, color='black')
		plt.ylim(0, 12)
		if model_dict[op_dict['model']]['model_classification'] == "binary":
			plt.xlim(0,1)
		elif model_dict[op_dict['model']]['model_classification'] == "regression":
			plt.xlim(0,12)

	# if a regression model, create a linear fit for the data and calculate R^2
	if model_dict[op_dict['model']]['model_classification'] == 'regression':

		# fit the data to a line
		fit = np.polyfit(x_prob, y_data, 1, full = True)
		p = np.poly1d(np.polyfit(x_prob, y_data, 1))
		x_line = np.linspace(np.amin(x_prob), np.amax(x_prob), 200)
		y_line = p(x_line)

		# calculate the linear r2 value for the linear regression
		# SSE = fit[1][0]
		# avg = 0.
		# for i in range(len(y_data)):
		# 	avg += y_data[i]
		# avg = avg / len(y_data)
		# SST = [y_data[i] - avg for i in range(len(y_data))]
		# for i in range(len(SST)):
		# 	SST[i] = SST[i] ** 2
		# SST_sum = 0.
		# for i in range(len(SST)):
		# 	SST_sum += SST[i]
		# R2 = 1 - SSE / SST_sum
		R2 = r2_score(y_data, p(x_prob))

		# add the line to the curve
		plt.plot(x_line, y_line, '--', color = 'r', label = "Linear Regression")
		plt.plot([], [], ' ', label="{0}".format(p))
		plt.plot([], [], ' ', label="R2 = {:.3f}".format(R2))
		plt.legend(loc = 'upper left')

	if model_dict[op_dict['model']]['model_classification'] == 'binary':
		# add plot labels
		plt.plot([0.5, 0.5], [0., 12.], color = 'black', linestyle = '--') 
		# line corresponding to model prediction of active vs. inactive
		plt.plot([0, 1.0], [6., 6.], color = 'black', linestyle = '--')
		# line corresponding to measured activity of active or inactive (based on model cutoff)

	plt.suptitle(title)
	plt.title(subtitle)
	plt.xlabel(x_axis_label.format())
	plt.ylabel(y_axis_label.format())

	# save results
	save_path = ""
	save_path = op_dict['save_to']	
	if op_dict['save_to'] == 'model_dir':
		save_path = model_config['model_dir']
	elif op_dict['save_to'] == 'data_dir':
		save_path = data_config['data_dir']
	save_path += data_name + "_" + model_name + "_predictions.tiff"
	plt.savefig(save_path, dpi = dpi_set)
	plt.show() 

def plot_truth_table(op_dict, data_dict, model_dict, set_dict):
	pass

	# load settings
	if verbose:
		print(f"\nPlotting Truth Table Predictions of DATASET ({op_dict['data']}) with MODEL ({op_dict['model']})")

	# load settings
	dpi_set = default_dpi
	if 'dpi' in set_dict:
		dpi_set = int(set_dict['dpi'])

	title = f"Predicted Activity of {op_dict['data']} vs. Measured Activity"
	if 'title' in op_dict['title']:
		title = op_dict['title']
	
	subtitle = f"(model = {op_dict['model']})"
	if 'subtitle' in op_dict:
		subtitle = op_dict['subtitle']

	x_axis_label = "Predicted Activity Greater than 6 NLM"
	if 'x_axis_label' in op_dict:
		x_axis_label = op_dict['x_axis_label']

	y_axis_label = "Measured Activity Greater than 6 NLM"
	if 'y_axis_label' in op_dict:
		y_axis_label = op_dict['y_axis_label']

	# load model
	model_name = op_dict['model'] # model name
	model_config = model_dict[model_name] # model config file
	model_path = model_config['model_dir'] + model_config['model_name']
	if verbose:
		print ("Loading model in: " + model_path)
	fp_type = model_config['fp_type'] # finger print used to create QSAR relationship
	model = joblib.load(model_path) # model generated by MegaModels package

	# load data
	data_name = op_dict['data'] # data set name
	data_config = data_dict[data_name] # data set config file
	data_path = data_config['data_dir'] + data_config['data_file']
	if verbose:
		print ("Loading data in: " + data_path)
	smiles_column = data_config['smile_column'] # column containing similes strings
	value_column = data_config['value_column'] # column containing values

	# get smile strings from data set, convert to fingerprints
	if verbose:
		print(f"Converting SMILE strings to {fp_type} ..")
	chem_fact = ChemFactoryLite() # object used to generate fingerprints from strings
	chem_fact.load_data(data_path, smiles_col=smiles_column, val_col=value_column)
	fp_data, y_data = chem_fact.get_data(Smiles=False, fp_type=fp_type)# if the data set has standard deviation values, pull them from the file
	y_err = []
	has_range = False
	if ('range_upper_column' in data_config) and ('range_lower_column' in data_config):
		has_range = True
		if verbose:
			print("Loading value range from: " + data_path)

		# create data frame containing std values
		data_df = pd.read_csv(data_path)
		y_low_range = data_df[data_config['range_lower_column']].tolist()
		y_up_range = data_df[data_config['range_upper_column']].tolist()

		# for the lower range values
		y_low_err = []
		for i in range(len(y_low_range)):
			# the lower bounds is the difference between the calculated value
			# and the lower bounds
			y_low_err.append(y_data[i] - y_low_range[i])

		# for the upper range values
		y_up_err = []
		for i in range(len(y_up_range)):
			# the upper bounds is the difference between the upper bounds and 
			# the calculated value
			y_up_err.append(y_up_range[i] - y_data[i])

		# calculate the range as the difference between the upper and lower values
		y_err = [y_low_err, y_up_err]

	# make predictions
	# NOTE :: this method assumes that the model is binary (ie not regression)
	# returns the probabilities that compound is either: [0] inactive, [1] active
	if verbose:
		print(f"Predicting data set values ..")
	probs = model.predict_proba(fp_data)
	# get the probability-like score of the compounds activity
	x_prob = []
	for k in range(len(probs)):
		x_prob.append(probs[k][1])

	# classify data
	# NOTE :: this program assumes that the cutoff is 6 NLM
	# override has_range for the time being
	has_range = False
	if has_range:
		# if a range is associated with the dataset
		predict = ["Yes", "No"]
		actual = ["Yes", "Maybe", "No"]
		# create data frame
		tt = pd.DataFrame(0, columns = predict, index = actual)

		# classify the compounds according to their measured and predicted acitivty
		for i in range(len(x_prob)):
			# classify the prediction
			predicted = (x_prob[i] > 0.5)

			# classify the measurement
			if ((y_data[i] > y_up_err[i]) and (y_data[i] > 6.)):
				# the actual classification is yes
				if predicted:
					tt.at["Yes", "Yes"] += 1
				else:
					tt.at["Yes", "No"] += 1
			elif ((y_data[i] < y_low_err[i]) and (y_data < 6.)):
				# the actual classification is no
				if predicted:
					tt.at["No", "Yes"] += 1
				else:
					tt.at["No", "No"] += 1
			else:
				# the actual classification is maybe
				if predicted:
					tt.at["Maybe", "Yes"] += 1
				else:
					tt.at["Maybe", "No"] += 1

		print(tt)
		exit()
	else:
		# if a range is associated with the dataset
		predict = ["Yes", "No"]
		actual = ["Yes", "No"]
		# create data frame
		tt = pd.DataFrame(0, columns = predict, index = actual)

		# classify the compounds according to their measured and predicted acitivty
		for i in range(len(x_prob)):
			# classify the prediction
			predicted = (x_prob[i] > 0.5)

			# classify the measurement
			if (y_data[i] > 6.):
				# the actual classification is yes
				if predicted:
					tt.at["Yes", "Yes"] += 1
				else:
					tt.at["Yes", "No"] += 1
			else:
				# the actual classification is no
				if predicted:
					tt.at["No", "Yes"] += 1
				else:
					tt.at["No", "No"] += 1

	# plot data
	sns.heatmap(tt, annot=True, linewidths=2, linecolor='black', cmap = 'Greens', fmt = 'g')
	plt.suptitle(title)
	plt.title(subtitle)
	plt.xlabel(x_axis_label.format())
	plt.ylabel(y_axis_label.format())

	save_path = ""
	save_path = op_dict['save_to']	
	if op_dict['save_to'] == 'model_dir':
		save_path = model_config['model_dir']
	elif op_dict['save_to'] == 'data_dir':
		save_path = data_config['data_dir']
	save_path += data_name + "_" + model_name + "_truth_table.tiff"
	plt.savefig(save_path, dpi = dpi_set)
	plt.show() 


