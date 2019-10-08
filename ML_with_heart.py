numerical_list = [‘resting_blood_pressure’, ‘serum_cholesterol_mg_per_dl’, ‘oldpeak_eq_st_depression’, ‘age’, ‘max_heart_rate_achieved’,‘slope_of_peak_exercise_st_segment’, ‘num_major_vessels’, ‘resting_ekg_results’]

categorical_list = [‘chest_pain_type’, ‘sex’, ‘exercise_induced_angina’, ‘fasting_blood_sugar_gt_120_mg_per_dl’, ‘thal’]

dummies_list = [‘chest_pain_type’,‘sex’,‘exercise_induced_angina’, ‘fasting_blood_sugar_gt_120_mg_per_dl’,‘thal_normal’, ‘thal_reversible_defect’, ‘thal_fixed_defect’]

values_file = pd.read_csv(‘train_values.csv’,header=0)
labels_file = pd.read_csv(‘train_labels.csv’,header=0)
no_pt_labels = labels_file.drop(‘patient_id’,axis=1)

categorical_values = values_file[categorical_list]
categorical_values = pd.get_dummies(categorical_values)

numerical_values = values_file[numerical_list]
train_std = numerical_values.std()
train_mean = numerical_values.mean()
cut_off = train_std * 3

lower, upper = train_mean - cut_off, train_mean + cut_off

trimmed_numerical_values = numerical_values[(numerical_values < upper) & (numerical_values > lower)]
numerical_df = pd.concat([trimmed_numerical_values,no_pt_labels,categorical_values],axis=1)
numerical_df = numerical_df.dropna()
numerical_df[dummies_list] = numerical_df[dummies_list].astype(‘category’)

numerical_labels = numerical_df[‘heart_disease_present’]
numerical_data = numerical_df.drop(‘heart_disease_present’,axis=1)

train_x, test_x, train_y, test_y = train_test_split(numerical_data,numerical_labels,test_size=0.15,random_state=7)

minmax = MinMaxScaler()
fitted = minmax.fit_transform(train_x)

fit_test_x = minmax.transform(test_x)

predictors = fitted
target = np.array(train_y)
test_pred = fit_test_x
test_target = np.array(test_y)

list_of_scores = []
val_scores = []
acc_scores = []
early_stop = EarlyStopping(patience=3)
epochs = range(1,100,5)
model = Sequential()
n_col = train_x.shape[1]
for number in epochs:
model.add(Dense(15, activation=‘relu’, input_shape=(n_col,)))
model.add(Dense(256, activation=‘relu’))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
test_target = np.array(test_y)
model.fit(predictors, target, epochs=number, verbose=False, validation_split=0.2,batch_size=18)
loss = model.evaluate(test_pred,test_target)[0]
print(model.evaluate(test_pred,test_target))
list_of_scores.append(loss)
meow = model.history
val_scores.append(meow.history['val_acc'][-1])
acc_scores.append(meow.history['acc'][-1])

history = model.history

plt.plot(list_of_scores)
plt.title(‘model accuracy’)
plt.ylabel(‘accuracy’)
plt.xlabel(‘epoch’)
plt.show()

print(model.evaluate(test_pred,test_target))
