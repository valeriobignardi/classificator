import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import ReviewDashboard from '../screens/ReviewDashboard';
import CasesList from '../screens/CasesList';
import CaseDetail from '../screens/CaseDetail';
import ClusteringParameters from '../screens/ClusteringParameters';
import LLMConfiguration from '../screens/LLMConfiguration';
import Scheduler from '../screens/Scheduler';
import TrainingFiles from '../screens/TrainingFiles';
import ReviewQueueThresholds from '../screens/ReviewQueueThresholds';

const ReviewStack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function ReviewStackNavigator() {
  return (
    <ReviewStack.Navigator>
      <ReviewStack.Screen name="ReviewDashboard" component={ReviewDashboard} options={{ title: 'Review' }} />
      <ReviewStack.Screen name="CasesList" component={CasesList} options={{ title: 'Casi' }} />
      <ReviewStack.Screen name="CaseDetail" component={CaseDetail} options={{ title: 'Dettagli Caso' }} />
    </ReviewStack.Navigator>
  );
}

export default function Tabs() {
  return (
    <Tab.Navigator screenOptions={{ headerShown: false }}>
      <Tab.Screen name="Review" component={ReviewStackNavigator} />
      <Tab.Screen name="Clustering" component={ClusteringParameters} />
      <Tab.Screen name="LLM" component={LLMConfiguration} />
      <Tab.Screen name="Scheduler" component={Scheduler} />
      <Tab.Screen name="Training" component={TrainingFiles} />
      <Tab.Screen name="Thresholds" component={ReviewQueueThresholds} />
    </Tab.Navigator>
  );
}
