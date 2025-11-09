import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaView, StatusBar, View } from 'react-native';
import { TenantProvider } from './src/contexts/TenantContext';
import Tabs from './src/navigation/Tabs';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <TenantProvider>
      <NavigationContainer>
        <StatusBar barStyle="dark-content" />
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          <Stack.Screen name="Root" component={Tabs} />
        </Stack.Navigator>
      </NavigationContainer>
    </TenantProvider>
  );
}

