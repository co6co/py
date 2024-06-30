// eslint-disable-next-line prettier/prettier
export const INSTALLED_KEY: unique symbol = Symbol('IPin')
export const PiniaInstanceKey: unique symbol = Symbol('PiniaInstanceKey')
const PERMISS_KEY = Symbol('PERMISS_KEY')
export const ConstObject = {
  [PERMISS_KEY]: 'permiss',
  getPermissValue: () => {
    return ConstObject[PERMISS_KEY]
  },
}
