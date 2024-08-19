export interface resourceOption {
	url: string;
	poster: string;
	/**
	 * 0: video
	 * 1. image
	 */
	type: 0 | 1; //0 video 1：image
	name: string;
	//是否需要认证
	authon?: boolean;
	posterAuthon?: boolean;
}

export type imageOption = Omit<resourceOption, 'type'>;
export type image2Option = Pick<resourceOption, 'url' | 'authon'>;
export type videoOption = Omit<resourceOption, 'type'>;
