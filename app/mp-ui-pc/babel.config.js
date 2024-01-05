//https://babeljs.io/docs/configuration
module.exports = function (api) {
    api.cache(true); 
    //const presets = [ ... ];
    const plugins = ["@vue/babel-plugin-jsx"];
    console.info("babel.config...")
    return {
     // presets,
      plugins
    };
  }