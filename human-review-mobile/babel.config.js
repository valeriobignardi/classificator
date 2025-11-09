module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // Necessario per @react-navigation e Bottom Tabs (expo compatibile)
      'react-native-reanimated/plugin',
    ],
  };
};
