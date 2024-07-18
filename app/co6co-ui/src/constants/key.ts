// eslint-disable-next-line prettier/prettier
export const INSTALLED_KEY: unique symbol = Symbol('IPin');
export const PiniaInstanceKey: unique symbol = Symbol('PiniaInstanceKey');
const PERMISS_KEY = Symbol('PERMISS_KEY');
const NONPERMISS_KEY = Symbol('NONPERMISS_KEY');
export const ConstObject = {
	[PERMISS_KEY]: 'permiss',
	[NONPERMISS_KEY]: 'nonPermiss',
	getPermissValue: () => {
		return ConstObject[PERMISS_KEY];
	},
	getNonPermissValue: () => {
		return ConstObject[NONPERMISS_KEY];
	},
};
