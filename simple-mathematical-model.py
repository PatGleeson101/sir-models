# Simple mathematical SIR model
#Imports
import math
import numpy as np
import time
import random as rand

#Simulation methods
class Population:
	def __init__(self, S, I, R = 0, N = None):
		self.S = S #Susceptible population
		self.I = I #Infected population
		self.R = R
		if (R == 0) and (N != None):
			self.R = N - I - S
		else:
			self.R = R #Removed population
		self.N = S + I + R #Total population size

class Infection:
	def __init__(self, beta, gamma):
		self.beta = beta #Effective contact rate
		self.gamma = gamma #Recovery rate
		self.R0 = beta / gamma #Basic reproduction number

class MetaData:
	def __init__(self, dt, dtype, duration, iteration_count, sample_count):
		self.dt = dt #Timestep
		self.dtype = dtype #Data type of stored values
		self.duration = duration #Virtual duration of simulation
		self.iteration_count = 0 #Number of iterations performed
		self.sample_count = 0 #Length of series (= iteration_count + 1)

class TimeSeriesData:
	def __init__(self, t_series, S_series, I_series, R_series):
		self.t_series = t_series #Time samples
		self.S_series = S_series #Susceptible population over time
		self.I_series = I_series #Infected population over time
		self.R_series = R_series #Removed population over time

class Model:
	def __init__(self, initial_population, infection, time_series_data, meta_data):
		self.initial_population = initial_population
		self.infection = infection
		self.time_series_data = time_series_data
		self.meta_data = meta_data

def simulate(initial_population, infection, dt, max_duration, transfer_rate_tolerance = 0):
	#Cache parameters
	N = initial_population.N
	S = initial_population.S
	I = initial_population.I
	R = initial_population.R
	beta = infection.beta
	gamma = infection.gamma
	#Determine internal simulation parameters
	#Data type
	dtype = None
	if (N < 6.55040e+04):
		dtype = np.float16
	elif (N < 3.4028235e+38):
		dtype = np.float32
	elif (N < 1.7976931348623157e+308):
		dtype = np.float64
	else:
		raise Exception("Population too large for floating-point datatype.")
	#Iteration limit
	max_iterations = math.ceil(max_duration / dt)
	max_samples = max_iterations + 1
	transfer_increment_tolerance = transfer_rate_tolerance * dt * N
	#Initialise storage
	S_series = np.zeros(max_samples, dtype = dtype) #Susceptible population time series
	I_series = np.zeros(max_samples, dtype = dtype) #Infected population time series
	R_series = np.zeros(max_samples, dtype = dtype) #Removed population time series
	t_series = np.zeros(max_samples, dtype = dtype) #Time data
	#Store initial values
	S_series[0] = S
	I_series[0] = I
	R_series[0] = R
	#t_series[0] = 0 #Redundant
	#Run simulation
	t = 0
	i = 0
	while i < max_iterations:
		#Increment iteration
		t += dt
		i += 1
		#Compute increments
		dS = - beta * I * S / N
		dR = gamma * I
		dI = - dS - dR
		#Update compartments
		S += dS * dt
		I += dI * dt
		R += dR * dt
		#Store current values
		S_series[i] = S
		I_series[i] = I
		R_series[i] = R
		t_series[i] = t
		#Check for equilibrium
		if (abs(dS) + abs(dR) < transfer_increment_tolerance):
			break
	#Remove excess storage
	t_series = t_series[:i]
	S_series = S_series[:i]
	I_series = I_series[:i]
	R_series = R_series[:i]
	#Return data
	time_series_data = TimeSeriesData(t_series, S_series, I_series, R_series)
	meta_data = MetaData(dt, dtype, t, i, i + 1)
	return Model(initial_population, infection, time_series_data, meta_data)

#Interactive demo
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
	
	#Default parameters
	default_population = Population(S = 990, I = 10, R = 0)
	default_infection = Infection(beta = 0.7, gamma = 0.2)
	default_max_duration = 100
	default_tolerance = 0.01
	default_dt = 0.05
	default_model = simulate(default_population, default_infection, default_dt, default_max_duration, default_tolerance)
	#Active model
	active_model = default_model #Current model
	active_population = default_population
	active_N = active_population.N
	active_infection = default_infection

	#Initialise matplotlib
	#Figure
	fig = plt.figure(figsize = (10, 6), dpi = 120) #Figure
	fig.suptitle("Simple mathematical SIR model") #Figure title
	fig.canvas.set_window_title("SIR model demo")
	fig.set_tight_layout({"rect": [0.05, 0.05, 0.95, 0.95]})

	#Gridspec layout
	gs_fig = fig.add_gridspec(ncols = 2, nrows = 2, height_ratios = [2, 1]) #Figure gridspec
	gs_widgets = gs_fig[1, 0:2].subgridspec(4, 2) #Widgets gridspec

	#Colours
	rgb_widget_ax = (0.5, 0.5, 0.5)
	rgb_S = (0.0, 0.0, 1.0)
	rgb_I = (1.0, 0.0, 0.0)
	rgb_R = (0.0, 0.0, 0.0)

	#Plots
	ax_time = fig.add_subplot(gs_fig[0, 0]); #Time series axes
	ax_phase = fig.add_subplot(gs_fig[0, 1]); #Phase plane axes

	#Phase plane
	ax_phase.set_xlabel('Susceptible (persons)')
	ax_phase.set_ylabel('Infected (persons)')
	ax_phase.set_title("Phase plane")
	line_phase, = ax_phase.plot([], [], "blue") #Current model as phase line
	quiv_phase = ax_phase.quiver([], [], [], [], [], pivot = 'mid', scale_units = 'xy') #Quiver arrows
	#fig.colorbar(quiv_phase, extend='max')

	#Time series plot
	ax_time.set_xlabel('Time (days)')
	ax_time.set_ylabel('Persons')
	ax_time.set_title("Time series")
	#Initialise lines
	line_S, = ax_time.plot([], [], color = rgb_S, label = 'S')
	line_I, = ax_time.plot([], [], color = rgb_I, label = 'I')
	line_R, = ax_time.plot([], [], color = rgb_R, label = 'R')
	#Legend
	ax_time.legend( (line_S, line_I, line_R), ("Susceptible", "Infected", "Removed") )

	#Widgets
	#Widget axes
	ax_beta = fig.add_subplot(gs_widgets[0, 0], facecolor = rgb_widget_ax)
	ax_gamma = fig.add_subplot(gs_widgets[1, 0], facecolor = rgb_widget_ax)
	ax_population = fig.add_subplot(gs_widgets[2, 0], facecolor = rgb_widget_ax)
	ax_update = fig.add_subplot(gs_widgets[3, 0], facecolor = rgb_widget_ax)
	ax_saveFig = fig.add_subplot(gs_widgets[0, 1], facecolor = rgb_widget_ax)
	ax_saveTimeAx = fig.add_subplot(gs_widgets[1, 1], facecolor = rgb_widget_ax)
	ax_savePhaseAx = fig.add_subplot(gs_widgets[2, 1], facecolor = rgb_widget_ax)
	#Sliders
	sld_beta= Slider(ax_beta, r'$\beta$', 0.1, 10.0, valinit = default_infection.beta)
	sld_gamma = Slider(ax_gamma, r'$\gamma$', 0.1, 10.0, valinit = default_infection.gamma)
	sld_population = Slider(ax_population, 'N', 10.0, 20000.0, valinit = active_N)
	#Buttons
	btn_update = Button(ax_update, 'Apply changes', color = rgb_widget_ax, hovercolor = '0.975')
	btn_saveFig = Button(ax_saveFig, 'Save figure', color = rgb_widget_ax, hovercolor = '0.975')
	btn_saveTimeAx = Button(ax_saveTimeAx, 'Save time series', color = rgb_widget_ax, hovercolor = '0.975')
	btn_savePhaseAx = Button(ax_savePhaseAx, 'Save phase plane', color = rgb_widget_ax, hovercolor = '0.975')
	#Instructions
	txt_instructions = fig.text(0.1, 0.9, 
	"""Click a point from the arrow region on the $\it{Phase \: plane}$ to select a starting population.\nThe simulation graph will appear on the left.""")

	#Functions
	def updateQuiver(N, infection):
		global quiv_phase
		#Generate new flow field quiver
		samplesS = np.linspace(0, N, 15)
		samplesI = np.linspace(0, N, 15)
		fieldS, fieldI = np.meshgrid(samplesS, samplesI)
		flowFieldS = -(infection.beta / N) * fieldS * fieldI
		flowFieldR = infection.gamma * fieldI
		flowFieldI = - flowFieldS - flowFieldR
		#Set invalid states (S + I > N) to zero
		flowFieldS = np.flip(np.tril(np.flip(flowFieldS, axis = 0), k = 0), axis = 0)
		flowFieldI = np.flip(np.tril(np.flip(flowFieldI, axis = 0), k = 0), axis = 0)
		#Generate transfer rate field (for colour mapping)
		rateField = np.abs(flowFieldS) + np.abs(flowFieldR)
		#Rescale flow fields to improve arrow sizing
		magnitudeField = np.hypot(flowFieldS, flowFieldI)
		targetMagnitudeField = np.log(magnitudeField + 1)
		scaleFactorField = np.divide(targetMagnitudeField, magnitudeField, out=np.zeros_like(targetMagnitudeField), where=(magnitudeField!=0)) 
		flowFieldS *= scaleFactorField
		flowFieldI *= scaleFactorField
		#Plot quiver
		quiv_phase.remove()
		quiv_phase = ax_phase.quiver(fieldS, fieldI, flowFieldS, flowFieldI, rateField, pivot = 'tail', scale_units = 'xy')

	def updateTimeSeries(model): #Update time series data
		global line_S, line_I, line_R, line_phase
		#Get data
		tsd = model.time_series_data
		t = tsd.t_series
		S = tsd.S_series
		I = tsd.I_series
		R = tsd.R_series
		#Plot data
		line_S.set_data(t, S)
		line_I.set_data(t, I)
		line_R.set_data(t, R)
		#Set time domain
		ax_time.set_xlim(0, model.meta_data.duration)
		#Line solution in phase space
		line_phase.set_data(S, I)

	def updateParams(val = None):
		global active_N, active_infection
		global line_phase, line_S, line_I, line_R
		#Update parameters
		active_N = sld_population.val
		active_infection = Infection(beta = sld_beta.val, gamma = sld_gamma.val)
		#Format time series vertical axis
		ax_time.set_ylim(0, active_N)
		#Format phase plane
		ax_phase.set_ylim(0, active_N)
		ax_phase.set_xlim(0, active_N)
		#Remove current quiver & lines
		line_phase.set_data([], [])
		line_S.set_data([], [])
		line_I.set_data([], [])
		line_R.set_data([], [])
		#Update quiver
		updateQuiver(active_N, active_infection)
		#Redraw
		fig.canvas.draw_idle()

	def updateModel(S, I):
		global active_N, active_population, active_infection
		active_population = Population(S = S, I = I, N = active_N)
		active_model = simulate(active_population, active_infection, default_dt, default_max_duration, default_tolerance)
		updateTimeSeries(active_model)
		fig.canvas.draw_idle()

	def onmouserelease(event):
		global active_model
		ex = event.xdata
		ey = event.ydata
		if (event.inaxes == ax_phase):
			if (ex > 0) and (ey > 0) and (0 <= ex + ey <= active_N):
				updateModel(ex, ey)

	def saveFig(event = None):
		t = time.localtime()
		fig.savefig(time.strftime('%Y-%m-%d %H.%M %Z.png', t))

	def saveTimeAx(event = None):
		t = time.localtime()
		extent = ax_time.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		fig.savefig(time.strftime('%Y-%m-%d %H.%M %Z.png', t), bbox_inches = extent.expanded(1.1, 1.2))

	def savePhaseAx(event = None):
		t = time.localtime()
		extent = ax_phase.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		fig.savefig(time.strftime('%Y-%m-%d %H.%M %Z.png', t), bbox_inches = extent.expanded(1.1, 1.2))

	#Connect interactivity
	#sld_beta.on_changed(updateParams)
	btn_update.on_clicked(updateParams)
	btn_saveFig.on_clicked(saveFig)
	btn_saveTimeAx.on_clicked(saveTimeAx)
	btn_savePhaseAx.on_clicked(savePhaseAx)
	cid_mouserelease = fig.canvas.mpl_connect('button_release_event', onmouserelease)

	#Initialise model
	updateParams()
	plt.show()