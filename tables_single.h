#ifndef TABLES__SINGLE_H
#define TABLES__SINGLE_H

#ifndef AESx
#define AESx(x) ((cl_uint)(x))
#endif

/*
 * The AES*[] tables allow us to perform a fast evaluation of an AES
 * round; table AESi[] combines SubBytes for a byte at row i, and
 * MixColumns for the column where that byte goes after ShiftRows.
 */

const cl_uint AES_c[1024] = {
	AESx(0xA56363C6), AESx(0x847C7CF8), AESx(0x997777EE), AESx(0x8D7B7BF6),
	AESx(0x0DF2F2FF), AESx(0xBD6B6BD6), AESx(0xB16F6FDE), AESx(0x54C5C591),
	AESx(0x50303060), AESx(0x03010102), AESx(0xA96767CE), AESx(0x7D2B2B56),
	AESx(0x19FEFEE7), AESx(0x62D7D7B5), AESx(0xE6ABAB4D), AESx(0x9A7676EC),
	AESx(0x45CACA8F), AESx(0x9D82821F), AESx(0x40C9C989), AESx(0x877D7DFA),
	AESx(0x15FAFAEF), AESx(0xEB5959B2), AESx(0xC947478E), AESx(0x0BF0F0FB),
	AESx(0xECADAD41), AESx(0x67D4D4B3), AESx(0xFDA2A25F), AESx(0xEAAFAF45),
	AESx(0xBF9C9C23), AESx(0xF7A4A453), AESx(0x967272E4), AESx(0x5BC0C09B),
	AESx(0xC2B7B775), AESx(0x1CFDFDE1), AESx(0xAE93933D), AESx(0x6A26264C),
	AESx(0x5A36366C), AESx(0x413F3F7E), AESx(0x02F7F7F5), AESx(0x4FCCCC83),
	AESx(0x5C343468), AESx(0xF4A5A551), AESx(0x34E5E5D1), AESx(0x08F1F1F9),
	AESx(0x937171E2), AESx(0x73D8D8AB), AESx(0x53313162), AESx(0x3F15152A),
	AESx(0x0C040408), AESx(0x52C7C795), AESx(0x65232346), AESx(0x5EC3C39D),
	AESx(0x28181830), AESx(0xA1969637), AESx(0x0F05050A), AESx(0xB59A9A2F),
	AESx(0x0907070E), AESx(0x36121224), AESx(0x9B80801B), AESx(0x3DE2E2DF),
	AESx(0x26EBEBCD), AESx(0x6927274E), AESx(0xCDB2B27F), AESx(0x9F7575EA),
	AESx(0x1B090912), AESx(0x9E83831D), AESx(0x742C2C58), AESx(0x2E1A1A34),
	AESx(0x2D1B1B36), AESx(0xB26E6EDC), AESx(0xEE5A5AB4), AESx(0xFBA0A05B),
	AESx(0xF65252A4), AESx(0x4D3B3B76), AESx(0x61D6D6B7), AESx(0xCEB3B37D),
	AESx(0x7B292952), AESx(0x3EE3E3DD), AESx(0x712F2F5E), AESx(0x97848413),
	AESx(0xF55353A6), AESx(0x68D1D1B9), AESx(0x00000000), AESx(0x2CEDEDC1),
	AESx(0x60202040), AESx(0x1FFCFCE3), AESx(0xC8B1B179), AESx(0xED5B5BB6),
	AESx(0xBE6A6AD4), AESx(0x46CBCB8D), AESx(0xD9BEBE67), AESx(0x4B393972),
	AESx(0xDE4A4A94), AESx(0xD44C4C98), AESx(0xE85858B0), AESx(0x4ACFCF85),
	AESx(0x6BD0D0BB), AESx(0x2AEFEFC5), AESx(0xE5AAAA4F), AESx(0x16FBFBED),
	AESx(0xC5434386), AESx(0xD74D4D9A), AESx(0x55333366), AESx(0x94858511),
	AESx(0xCF45458A), AESx(0x10F9F9E9), AESx(0x06020204), AESx(0x817F7FFE),
	AESx(0xF05050A0), AESx(0x443C3C78), AESx(0xBA9F9F25), AESx(0xE3A8A84B),
	AESx(0xF35151A2), AESx(0xFEA3A35D), AESx(0xC0404080), AESx(0x8A8F8F05),
	AESx(0xAD92923F), AESx(0xBC9D9D21), AESx(0x48383870), AESx(0x04F5F5F1),
	AESx(0xDFBCBC63), AESx(0xC1B6B677), AESx(0x75DADAAF), AESx(0x63212142),
	AESx(0x30101020), AESx(0x1AFFFFE5), AESx(0x0EF3F3FD), AESx(0x6DD2D2BF),
	AESx(0x4CCDCD81), AESx(0x140C0C18), AESx(0x35131326), AESx(0x2FECECC3),
	AESx(0xE15F5FBE), AESx(0xA2979735), AESx(0xCC444488), AESx(0x3917172E),
	AESx(0x57C4C493), AESx(0xF2A7A755), AESx(0x827E7EFC), AESx(0x473D3D7A),
	AESx(0xAC6464C8), AESx(0xE75D5DBA), AESx(0x2B191932), AESx(0x957373E6),
	AESx(0xA06060C0), AESx(0x98818119), AESx(0xD14F4F9E), AESx(0x7FDCDCA3),
	AESx(0x66222244), AESx(0x7E2A2A54), AESx(0xAB90903B), AESx(0x8388880B),
	AESx(0xCA46468C), AESx(0x29EEEEC7), AESx(0xD3B8B86B), AESx(0x3C141428),
	AESx(0x79DEDEA7), AESx(0xE25E5EBC), AESx(0x1D0B0B16), AESx(0x76DBDBAD),
	AESx(0x3BE0E0DB), AESx(0x56323264), AESx(0x4E3A3A74), AESx(0x1E0A0A14),
	AESx(0xDB494992), AESx(0x0A06060C), AESx(0x6C242448), AESx(0xE45C5CB8),
	AESx(0x5DC2C29F), AESx(0x6ED3D3BD), AESx(0xEFACAC43), AESx(0xA66262C4),
	AESx(0xA8919139), AESx(0xA4959531), AESx(0x37E4E4D3), AESx(0x8B7979F2),
	AESx(0x32E7E7D5), AESx(0x43C8C88B), AESx(0x5937376E), AESx(0xB76D6DDA),
	AESx(0x8C8D8D01), AESx(0x64D5D5B1), AESx(0xD24E4E9C), AESx(0xE0A9A949),
	AESx(0xB46C6CD8), AESx(0xFA5656AC), AESx(0x07F4F4F3), AESx(0x25EAEACF),
	AESx(0xAF6565CA), AESx(0x8E7A7AF4), AESx(0xE9AEAE47), AESx(0x18080810),
	AESx(0xD5BABA6F), AESx(0x887878F0), AESx(0x6F25254A), AESx(0x722E2E5C),
	AESx(0x241C1C38), AESx(0xF1A6A657), AESx(0xC7B4B473), AESx(0x51C6C697),
	AESx(0x23E8E8CB), AESx(0x7CDDDDA1), AESx(0x9C7474E8), AESx(0x211F1F3E),
	AESx(0xDD4B4B96), AESx(0xDCBDBD61), AESx(0x868B8B0D), AESx(0x858A8A0F),
	AESx(0x907070E0), AESx(0x423E3E7C), AESx(0xC4B5B571), AESx(0xAA6666CC),
	AESx(0xD8484890), AESx(0x05030306), AESx(0x01F6F6F7), AESx(0x120E0E1C),
	AESx(0xA36161C2), AESx(0x5F35356A), AESx(0xF95757AE), AESx(0xD0B9B969),
	AESx(0x91868617), AESx(0x58C1C199), AESx(0x271D1D3A), AESx(0xB99E9E27),
	AESx(0x38E1E1D9), AESx(0x13F8F8EB), AESx(0xB398982B), AESx(0x33111122),
	AESx(0xBB6969D2), AESx(0x70D9D9A9), AESx(0x898E8E07), AESx(0xA7949433),
	AESx(0xB69B9B2D), AESx(0x221E1E3C), AESx(0x92878715), AESx(0x20E9E9C9),
	AESx(0x49CECE87), AESx(0xFF5555AA), AESx(0x78282850), AESx(0x7ADFDFA5),
	AESx(0x8F8C8C03), AESx(0xF8A1A159), AESx(0x80898909), AESx(0x170D0D1A),
	AESx(0xDABFBF65), AESx(0x31E6E6D7), AESx(0xC6424284), AESx(0xB86868D0),
	AESx(0xC3414182), AESx(0xB0999929), AESx(0x772D2D5A), AESx(0x110F0F1E),
	AESx(0xCBB0B07B), AESx(0xFC5454A8), AESx(0xD6BBBB6D), AESx(0x3A16162C),
	AESx(0x6363C6A5), AESx(0x7C7CF884), AESx(0x7777EE99), AESx(0x7B7BF68D),
	AESx(0xF2F2FF0D), AESx(0x6B6BD6BD), AESx(0x6F6FDEB1), AESx(0xC5C59154),
	AESx(0x30306050), AESx(0x01010203), AESx(0x6767CEA9), AESx(0x2B2B567D),
	AESx(0xFEFEE719), AESx(0xD7D7B562), AESx(0xABAB4DE6), AESx(0x7676EC9A),
	AESx(0xCACA8F45), AESx(0x82821F9D), AESx(0xC9C98940), AESx(0x7D7DFA87),
	AESx(0xFAFAEF15), AESx(0x5959B2EB), AESx(0x47478EC9), AESx(0xF0F0FB0B),
	AESx(0xADAD41EC), AESx(0xD4D4B367), AESx(0xA2A25FFD), AESx(0xAFAF45EA),
	AESx(0x9C9C23BF), AESx(0xA4A453F7), AESx(0x7272E496), AESx(0xC0C09B5B),
	AESx(0xB7B775C2), AESx(0xFDFDE11C), AESx(0x93933DAE), AESx(0x26264C6A),
	AESx(0x36366C5A), AESx(0x3F3F7E41), AESx(0xF7F7F502), AESx(0xCCCC834F),
	AESx(0x3434685C), AESx(0xA5A551F4), AESx(0xE5E5D134), AESx(0xF1F1F908),
	AESx(0x7171E293), AESx(0xD8D8AB73), AESx(0x31316253), AESx(0x15152A3F),
	AESx(0x0404080C), AESx(0xC7C79552), AESx(0x23234665), AESx(0xC3C39D5E),
	AESx(0x18183028), AESx(0x969637A1), AESx(0x05050A0F), AESx(0x9A9A2FB5),
	AESx(0x07070E09), AESx(0x12122436), AESx(0x80801B9B), AESx(0xE2E2DF3D),
	AESx(0xEBEBCD26), AESx(0x27274E69), AESx(0xB2B27FCD), AESx(0x7575EA9F),
	AESx(0x0909121B), AESx(0x83831D9E), AESx(0x2C2C5874), AESx(0x1A1A342E),
	AESx(0x1B1B362D), AESx(0x6E6EDCB2), AESx(0x5A5AB4EE), AESx(0xA0A05BFB),
	AESx(0x5252A4F6), AESx(0x3B3B764D), AESx(0xD6D6B761), AESx(0xB3B37DCE),
	AESx(0x2929527B), AESx(0xE3E3DD3E), AESx(0x2F2F5E71), AESx(0x84841397),
	AESx(0x5353A6F5), AESx(0xD1D1B968), AESx(0x00000000), AESx(0xEDEDC12C),
	AESx(0x20204060), AESx(0xFCFCE31F), AESx(0xB1B179C8), AESx(0x5B5BB6ED),
	AESx(0x6A6AD4BE), AESx(0xCBCB8D46), AESx(0xBEBE67D9), AESx(0x3939724B),
	AESx(0x4A4A94DE), AESx(0x4C4C98D4), AESx(0x5858B0E8), AESx(0xCFCF854A),
	AESx(0xD0D0BB6B), AESx(0xEFEFC52A), AESx(0xAAAA4FE5), AESx(0xFBFBED16),
	AESx(0x434386C5), AESx(0x4D4D9AD7), AESx(0x33336655), AESx(0x85851194),
	AESx(0x45458ACF), AESx(0xF9F9E910), AESx(0x02020406), AESx(0x7F7FFE81),
	AESx(0x5050A0F0), AESx(0x3C3C7844), AESx(0x9F9F25BA), AESx(0xA8A84BE3),
	AESx(0x5151A2F3), AESx(0xA3A35DFE), AESx(0x404080C0), AESx(0x8F8F058A),
	AESx(0x92923FAD), AESx(0x9D9D21BC), AESx(0x38387048), AESx(0xF5F5F104),
	AESx(0xBCBC63DF), AESx(0xB6B677C1), AESx(0xDADAAF75), AESx(0x21214263),
	AESx(0x10102030), AESx(0xFFFFE51A), AESx(0xF3F3FD0E), AESx(0xD2D2BF6D),
	AESx(0xCDCD814C), AESx(0x0C0C1814), AESx(0x13132635), AESx(0xECECC32F),
	AESx(0x5F5FBEE1), AESx(0x979735A2), AESx(0x444488CC), AESx(0x17172E39),
	AESx(0xC4C49357), AESx(0xA7A755F2), AESx(0x7E7EFC82), AESx(0x3D3D7A47),
	AESx(0x6464C8AC), AESx(0x5D5DBAE7), AESx(0x1919322B), AESx(0x7373E695),
	AESx(0x6060C0A0), AESx(0x81811998), AESx(0x4F4F9ED1), AESx(0xDCDCA37F),
	AESx(0x22224466), AESx(0x2A2A547E), AESx(0x90903BAB), AESx(0x88880B83),
	AESx(0x46468CCA), AESx(0xEEEEC729), AESx(0xB8B86BD3), AESx(0x1414283C),
	AESx(0xDEDEA779), AESx(0x5E5EBCE2), AESx(0x0B0B161D), AESx(0xDBDBAD76),
	AESx(0xE0E0DB3B), AESx(0x32326456), AESx(0x3A3A744E), AESx(0x0A0A141E),
	AESx(0x494992DB), AESx(0x06060C0A), AESx(0x2424486C), AESx(0x5C5CB8E4),
	AESx(0xC2C29F5D), AESx(0xD3D3BD6E), AESx(0xACAC43EF), AESx(0x6262C4A6),
	AESx(0x919139A8), AESx(0x959531A4), AESx(0xE4E4D337), AESx(0x7979F28B),
	AESx(0xE7E7D532), AESx(0xC8C88B43), AESx(0x37376E59), AESx(0x6D6DDAB7),
	AESx(0x8D8D018C), AESx(0xD5D5B164), AESx(0x4E4E9CD2), AESx(0xA9A949E0),
	AESx(0x6C6CD8B4), AESx(0x5656ACFA), AESx(0xF4F4F307), AESx(0xEAEACF25),
	AESx(0x6565CAAF), AESx(0x7A7AF48E), AESx(0xAEAE47E9), AESx(0x08081018),
	AESx(0xBABA6FD5), AESx(0x7878F088), AESx(0x25254A6F), AESx(0x2E2E5C72),
	AESx(0x1C1C3824), AESx(0xA6A657F1), AESx(0xB4B473C7), AESx(0xC6C69751),
	AESx(0xE8E8CB23), AESx(0xDDDDA17C), AESx(0x7474E89C), AESx(0x1F1F3E21),
	AESx(0x4B4B96DD), AESx(0xBDBD61DC), AESx(0x8B8B0D86), AESx(0x8A8A0F85),
	AESx(0x7070E090), AESx(0x3E3E7C42), AESx(0xB5B571C4), AESx(0x6666CCAA),
	AESx(0x484890D8), AESx(0x03030605), AESx(0xF6F6F701), AESx(0x0E0E1C12),
	AESx(0x6161C2A3), AESx(0x35356A5F), AESx(0x5757AEF9), AESx(0xB9B969D0),
	AESx(0x86861791), AESx(0xC1C19958), AESx(0x1D1D3A27), AESx(0x9E9E27B9),
	AESx(0xE1E1D938), AESx(0xF8F8EB13), AESx(0x98982BB3), AESx(0x11112233),
	AESx(0x6969D2BB), AESx(0xD9D9A970), AESx(0x8E8E0789), AESx(0x949433A7),
	AESx(0x9B9B2DB6), AESx(0x1E1E3C22), AESx(0x87871592), AESx(0xE9E9C920),
	AESx(0xCECE8749), AESx(0x5555AAFF), AESx(0x28285078), AESx(0xDFDFA57A),
	AESx(0x8C8C038F), AESx(0xA1A159F8), AESx(0x89890980), AESx(0x0D0D1A17),
	AESx(0xBFBF65DA), AESx(0xE6E6D731), AESx(0x424284C6), AESx(0x6868D0B8),
	AESx(0x414182C3), AESx(0x999929B0), AESx(0x2D2D5A77), AESx(0x0F0F1E11),
	AESx(0xB0B07BCB), AESx(0x5454A8FC), AESx(0xBBBB6DD6), AESx(0x16162C3A),
	AESx(0x63C6A563), AESx(0x7CF8847C), AESx(0x77EE9977), AESx(0x7BF68D7B),
	AESx(0xF2FF0DF2), AESx(0x6BD6BD6B), AESx(0x6FDEB16F), AESx(0xC59154C5),
	AESx(0x30605030), AESx(0x01020301), AESx(0x67CEA967), AESx(0x2B567D2B),
	AESx(0xFEE719FE), AESx(0xD7B562D7), AESx(0xAB4DE6AB), AESx(0x76EC9A76),
	AESx(0xCA8F45CA), AESx(0x821F9D82), AESx(0xC98940C9), AESx(0x7DFA877D),
	AESx(0xFAEF15FA), AESx(0x59B2EB59), AESx(0x478EC947), AESx(0xF0FB0BF0),
	AESx(0xAD41ECAD), AESx(0xD4B367D4), AESx(0xA25FFDA2), AESx(0xAF45EAAF),
	AESx(0x9C23BF9C), AESx(0xA453F7A4), AESx(0x72E49672), AESx(0xC09B5BC0),
	AESx(0xB775C2B7), AESx(0xFDE11CFD), AESx(0x933DAE93), AESx(0x264C6A26),
	AESx(0x366C5A36), AESx(0x3F7E413F), AESx(0xF7F502F7), AESx(0xCC834FCC),
	AESx(0x34685C34), AESx(0xA551F4A5), AESx(0xE5D134E5), AESx(0xF1F908F1),
	AESx(0x71E29371), AESx(0xD8AB73D8), AESx(0x31625331), AESx(0x152A3F15),
	AESx(0x04080C04), AESx(0xC79552C7), AESx(0x23466523), AESx(0xC39D5EC3),
	AESx(0x18302818), AESx(0x9637A196), AESx(0x050A0F05), AESx(0x9A2FB59A),
	AESx(0x070E0907), AESx(0x12243612), AESx(0x801B9B80), AESx(0xE2DF3DE2),
	AESx(0xEBCD26EB), AESx(0x274E6927), AESx(0xB27FCDB2), AESx(0x75EA9F75),
	AESx(0x09121B09), AESx(0x831D9E83), AESx(0x2C58742C), AESx(0x1A342E1A),
	AESx(0x1B362D1B), AESx(0x6EDCB26E), AESx(0x5AB4EE5A), AESx(0xA05BFBA0),
	AESx(0x52A4F652), AESx(0x3B764D3B), AESx(0xD6B761D6), AESx(0xB37DCEB3),
	AESx(0x29527B29), AESx(0xE3DD3EE3), AESx(0x2F5E712F), AESx(0x84139784),
	AESx(0x53A6F553), AESx(0xD1B968D1), AESx(0x00000000), AESx(0xEDC12CED),
	AESx(0x20406020), AESx(0xFCE31FFC), AESx(0xB179C8B1), AESx(0x5BB6ED5B),
	AESx(0x6AD4BE6A), AESx(0xCB8D46CB), AESx(0xBE67D9BE), AESx(0x39724B39),
	AESx(0x4A94DE4A), AESx(0x4C98D44C), AESx(0x58B0E858), AESx(0xCF854ACF),
	AESx(0xD0BB6BD0), AESx(0xEFC52AEF), AESx(0xAA4FE5AA), AESx(0xFBED16FB),
	AESx(0x4386C543), AESx(0x4D9AD74D), AESx(0x33665533), AESx(0x85119485),
	AESx(0x458ACF45), AESx(0xF9E910F9), AESx(0x02040602), AESx(0x7FFE817F),
	AESx(0x50A0F050), AESx(0x3C78443C), AESx(0x9F25BA9F), AESx(0xA84BE3A8),
	AESx(0x51A2F351), AESx(0xA35DFEA3), AESx(0x4080C040), AESx(0x8F058A8F),
	AESx(0x923FAD92), AESx(0x9D21BC9D), AESx(0x38704838), AESx(0xF5F104F5),
	AESx(0xBC63DFBC), AESx(0xB677C1B6), AESx(0xDAAF75DA), AESx(0x21426321),
	AESx(0x10203010), AESx(0xFFE51AFF), AESx(0xF3FD0EF3), AESx(0xD2BF6DD2),
	AESx(0xCD814CCD), AESx(0x0C18140C), AESx(0x13263513), AESx(0xECC32FEC),
	AESx(0x5FBEE15F), AESx(0x9735A297), AESx(0x4488CC44), AESx(0x172E3917),
	AESx(0xC49357C4), AESx(0xA755F2A7), AESx(0x7EFC827E), AESx(0x3D7A473D),
	AESx(0x64C8AC64), AESx(0x5DBAE75D), AESx(0x19322B19), AESx(0x73E69573),
	AESx(0x60C0A060), AESx(0x81199881), AESx(0x4F9ED14F), AESx(0xDCA37FDC),
	AESx(0x22446622), AESx(0x2A547E2A), AESx(0x903BAB90), AESx(0x880B8388),
	AESx(0x468CCA46), AESx(0xEEC729EE), AESx(0xB86BD3B8), AESx(0x14283C14),
	AESx(0xDEA779DE), AESx(0x5EBCE25E), AESx(0x0B161D0B), AESx(0xDBAD76DB),
	AESx(0xE0DB3BE0), AESx(0x32645632), AESx(0x3A744E3A), AESx(0x0A141E0A),
	AESx(0x4992DB49), AESx(0x060C0A06), AESx(0x24486C24), AESx(0x5CB8E45C),
	AESx(0xC29F5DC2), AESx(0xD3BD6ED3), AESx(0xAC43EFAC), AESx(0x62C4A662),
	AESx(0x9139A891), AESx(0x9531A495), AESx(0xE4D337E4), AESx(0x79F28B79),
	AESx(0xE7D532E7), AESx(0xC88B43C8), AESx(0x376E5937), AESx(0x6DDAB76D),
	AESx(0x8D018C8D), AESx(0xD5B164D5), AESx(0x4E9CD24E), AESx(0xA949E0A9),
	AESx(0x6CD8B46C), AESx(0x56ACFA56), AESx(0xF4F307F4), AESx(0xEACF25EA),
	AESx(0x65CAAF65), AESx(0x7AF48E7A), AESx(0xAE47E9AE), AESx(0x08101808),
	AESx(0xBA6FD5BA), AESx(0x78F08878), AESx(0x254A6F25), AESx(0x2E5C722E),
	AESx(0x1C38241C), AESx(0xA657F1A6), AESx(0xB473C7B4), AESx(0xC69751C6),
	AESx(0xE8CB23E8), AESx(0xDDA17CDD), AESx(0x74E89C74), AESx(0x1F3E211F),
	AESx(0x4B96DD4B), AESx(0xBD61DCBD), AESx(0x8B0D868B), AESx(0x8A0F858A),
	AESx(0x70E09070), AESx(0x3E7C423E), AESx(0xB571C4B5), AESx(0x66CCAA66),
	AESx(0x4890D848), AESx(0x03060503), AESx(0xF6F701F6), AESx(0x0E1C120E),
	AESx(0x61C2A361), AESx(0x356A5F35), AESx(0x57AEF957), AESx(0xB969D0B9),
	AESx(0x86179186), AESx(0xC19958C1), AESx(0x1D3A271D), AESx(0x9E27B99E),
	AESx(0xE1D938E1), AESx(0xF8EB13F8), AESx(0x982BB398), AESx(0x11223311),
	AESx(0x69D2BB69), AESx(0xD9A970D9), AESx(0x8E07898E), AESx(0x9433A794),
	AESx(0x9B2DB69B), AESx(0x1E3C221E), AESx(0x87159287), AESx(0xE9C920E9),
	AESx(0xCE8749CE), AESx(0x55AAFF55), AESx(0x28507828), AESx(0xDFA57ADF),
	AESx(0x8C038F8C), AESx(0xA159F8A1), AESx(0x89098089), AESx(0x0D1A170D),
	AESx(0xBF65DABF), AESx(0xE6D731E6), AESx(0x4284C642), AESx(0x68D0B868),
	AESx(0x4182C341), AESx(0x9929B099), AESx(0x2D5A772D), AESx(0x0F1E110F),
	AESx(0xB07BCBB0), AESx(0x54A8FC54), AESx(0xBB6DD6BB), AESx(0x162C3A16),
	AESx(0xC6A56363), AESx(0xF8847C7C), AESx(0xEE997777), AESx(0xF68D7B7B),
	AESx(0xFF0DF2F2), AESx(0xD6BD6B6B), AESx(0xDEB16F6F), AESx(0x9154C5C5),
	AESx(0x60503030), AESx(0x02030101), AESx(0xCEA96767), AESx(0x567D2B2B),
	AESx(0xE719FEFE), AESx(0xB562D7D7), AESx(0x4DE6ABAB), AESx(0xEC9A7676),
	AESx(0x8F45CACA), AESx(0x1F9D8282), AESx(0x8940C9C9), AESx(0xFA877D7D),
	AESx(0xEF15FAFA), AESx(0xB2EB5959), AESx(0x8EC94747), AESx(0xFB0BF0F0),
	AESx(0x41ECADAD), AESx(0xB367D4D4), AESx(0x5FFDA2A2), AESx(0x45EAAFAF),
	AESx(0x23BF9C9C), AESx(0x53F7A4A4), AESx(0xE4967272), AESx(0x9B5BC0C0),
	AESx(0x75C2B7B7), AESx(0xE11CFDFD), AESx(0x3DAE9393), AESx(0x4C6A2626),
	AESx(0x6C5A3636), AESx(0x7E413F3F), AESx(0xF502F7F7), AESx(0x834FCCCC),
	AESx(0x685C3434), AESx(0x51F4A5A5), AESx(0xD134E5E5), AESx(0xF908F1F1),
	AESx(0xE2937171), AESx(0xAB73D8D8), AESx(0x62533131), AESx(0x2A3F1515),
	AESx(0x080C0404), AESx(0x9552C7C7), AESx(0x46652323), AESx(0x9D5EC3C3),
	AESx(0x30281818), AESx(0x37A19696), AESx(0x0A0F0505), AESx(0x2FB59A9A),
	AESx(0x0E090707), AESx(0x24361212), AESx(0x1B9B8080), AESx(0xDF3DE2E2),
	AESx(0xCD26EBEB), AESx(0x4E692727), AESx(0x7FCDB2B2), AESx(0xEA9F7575),
	AESx(0x121B0909), AESx(0x1D9E8383), AESx(0x58742C2C), AESx(0x342E1A1A),
	AESx(0x362D1B1B), AESx(0xDCB26E6E), AESx(0xB4EE5A5A), AESx(0x5BFBA0A0),
	AESx(0xA4F65252), AESx(0x764D3B3B), AESx(0xB761D6D6), AESx(0x7DCEB3B3),
	AESx(0x527B2929), AESx(0xDD3EE3E3), AESx(0x5E712F2F), AESx(0x13978484),
	AESx(0xA6F55353), AESx(0xB968D1D1), AESx(0x00000000), AESx(0xC12CEDED),
	AESx(0x40602020), AESx(0xE31FFCFC), AESx(0x79C8B1B1), AESx(0xB6ED5B5B),
	AESx(0xD4BE6A6A), AESx(0x8D46CBCB), AESx(0x67D9BEBE), AESx(0x724B3939),
	AESx(0x94DE4A4A), AESx(0x98D44C4C), AESx(0xB0E85858), AESx(0x854ACFCF),
	AESx(0xBB6BD0D0), AESx(0xC52AEFEF), AESx(0x4FE5AAAA), AESx(0xED16FBFB),
	AESx(0x86C54343), AESx(0x9AD74D4D), AESx(0x66553333), AESx(0x11948585),
	AESx(0x8ACF4545), AESx(0xE910F9F9), AESx(0x04060202), AESx(0xFE817F7F),
	AESx(0xA0F05050), AESx(0x78443C3C), AESx(0x25BA9F9F), AESx(0x4BE3A8A8),
	AESx(0xA2F35151), AESx(0x5DFEA3A3), AESx(0x80C04040), AESx(0x058A8F8F),
	AESx(0x3FAD9292), AESx(0x21BC9D9D), AESx(0x70483838), AESx(0xF104F5F5),
	AESx(0x63DFBCBC), AESx(0x77C1B6B6), AESx(0xAF75DADA), AESx(0x42632121),
	AESx(0x20301010), AESx(0xE51AFFFF), AESx(0xFD0EF3F3), AESx(0xBF6DD2D2),
	AESx(0x814CCDCD), AESx(0x18140C0C), AESx(0x26351313), AESx(0xC32FECEC),
	AESx(0xBEE15F5F), AESx(0x35A29797), AESx(0x88CC4444), AESx(0x2E391717),
	AESx(0x9357C4C4), AESx(0x55F2A7A7), AESx(0xFC827E7E), AESx(0x7A473D3D),
	AESx(0xC8AC6464), AESx(0xBAE75D5D), AESx(0x322B1919), AESx(0xE6957373),
	AESx(0xC0A06060), AESx(0x19988181), AESx(0x9ED14F4F), AESx(0xA37FDCDC),
	AESx(0x44662222), AESx(0x547E2A2A), AESx(0x3BAB9090), AESx(0x0B838888),
	AESx(0x8CCA4646), AESx(0xC729EEEE), AESx(0x6BD3B8B8), AESx(0x283C1414),
	AESx(0xA779DEDE), AESx(0xBCE25E5E), AESx(0x161D0B0B), AESx(0xAD76DBDB),
	AESx(0xDB3BE0E0), AESx(0x64563232), AESx(0x744E3A3A), AESx(0x141E0A0A),
	AESx(0x92DB4949), AESx(0x0C0A0606), AESx(0x486C2424), AESx(0xB8E45C5C),
	AESx(0x9F5DC2C2), AESx(0xBD6ED3D3), AESx(0x43EFACAC), AESx(0xC4A66262),
	AESx(0x39A89191), AESx(0x31A49595), AESx(0xD337E4E4), AESx(0xF28B7979),
	AESx(0xD532E7E7), AESx(0x8B43C8C8), AESx(0x6E593737), AESx(0xDAB76D6D),
	AESx(0x018C8D8D), AESx(0xB164D5D5), AESx(0x9CD24E4E), AESx(0x49E0A9A9),
	AESx(0xD8B46C6C), AESx(0xACFA5656), AESx(0xF307F4F4), AESx(0xCF25EAEA),
	AESx(0xCAAF6565), AESx(0xF48E7A7A), AESx(0x47E9AEAE), AESx(0x10180808),
	AESx(0x6FD5BABA), AESx(0xF0887878), AESx(0x4A6F2525), AESx(0x5C722E2E),
	AESx(0x38241C1C), AESx(0x57F1A6A6), AESx(0x73C7B4B4), AESx(0x9751C6C6),
	AESx(0xCB23E8E8), AESx(0xA17CDDDD), AESx(0xE89C7474), AESx(0x3E211F1F),
	AESx(0x96DD4B4B), AESx(0x61DCBDBD), AESx(0x0D868B8B), AESx(0x0F858A8A),
	AESx(0xE0907070), AESx(0x7C423E3E), AESx(0x71C4B5B5), AESx(0xCCAA6666),
	AESx(0x90D84848), AESx(0x06050303), AESx(0xF701F6F6), AESx(0x1C120E0E),
	AESx(0xC2A36161), AESx(0x6A5F3535), AESx(0xAEF95757), AESx(0x69D0B9B9),
	AESx(0x17918686), AESx(0x9958C1C1), AESx(0x3A271D1D), AESx(0x27B99E9E),
	AESx(0xD938E1E1), AESx(0xEB13F8F8), AESx(0x2BB39898), AESx(0x22331111),
	AESx(0xD2BB6969), AESx(0xA970D9D9), AESx(0x07898E8E), AESx(0x33A79494),
	AESx(0x2DB69B9B), AESx(0x3C221E1E), AESx(0x15928787), AESx(0xC920E9E9),
	AESx(0x8749CECE), AESx(0xAAFF5555), AESx(0x50782828), AESx(0xA57ADFDF),
	AESx(0x038F8C8C), AESx(0x59F8A1A1), AESx(0x09808989), AESx(0x1A170D0D),
	AESx(0x65DABFBF), AESx(0xD731E6E6), AESx(0x84C64242), AESx(0xD0B86868),
	AESx(0x82C34141), AESx(0x29B09999), AESx(0x5A772D2D), AESx(0x1E110F0F),
	AESx(0x7BCBB0B0), AESx(0xA8FC5454), AESx(0x6DD6BBBB), AESx(0x2C3A1616)
};

const cl_uint mixtab_c[] = {
	(0x63633297), (0x7c7c6feb), (0x77775ec7),
	(0x7b7b7af7), (0xf2f2e8e5), (0x6b6b0ab7),
	(0x6f6f16a7), (0xc5c56d39), (0x303090c0),
	(0x01010704), (0x67672e87), (0x2b2bd1ac),
	(0xfefeccd5), (0xd7d71371), (0xabab7c9a),
	(0x767659c3), (0xcaca4005), (0x8282a33e),
	(0xc9c94909), (0x7d7d68ef), (0xfafad0c5),
	(0x5959947f), (0x4747ce07), (0xf0f0e6ed),
	(0xadad6e82), (0xd4d41a7d), (0xa2a243be),
	(0xafaf608a), (0x9c9cf946), (0xa4a451a6),
	(0x727245d3), (0xc0c0762d), (0xb7b728ea),
	(0xfdfdc5d9), (0x9393d47a), (0x2626f298),
	(0x363682d8), (0x3f3fbdfc), (0xf7f7f3f1),
	(0xcccc521d), (0x34348cd0), (0xa5a556a2),
	(0xe5e58db9), (0xf1f1e1e9), (0x71714cdf),
	(0xd8d83e4d), (0x313197c4), (0x15156b54),
	(0x04041c10), (0xc7c76331), (0x2323e98c),
	(0xc3c37f21), (0x18184860), (0x9696cf6e),
	(0x05051b14), (0x9a9aeb5e), (0x0707151c),
	(0x12127e48), (0x8080ad36), (0xe2e298a5),
	(0xebeba781), (0x2727f59c), (0xb2b233fe),
	(0x757550cf), (0x09093f24), (0x8383a43a),
	(0x2c2cc4b0), (0x1a1a4668), (0x1b1b416c),
	(0x6e6e11a3), (0x5a5a9d73), (0xa0a04db6),
	(0x5252a553), (0x3b3ba1ec), (0xd6d61475),
	(0xb3b334fa), (0x2929dfa4), (0xe3e39fa1),
	(0x2f2fcdbc), (0x8484b126), (0x5353a257),
	(0xd1d10169), (0x00000000), (0xededb599),
	(0x2020e080), (0xfcfcc2dd), (0xb1b13af2),
	(0x5b5b9a77), (0x6a6a0db3), (0xcbcb4701),
	(0xbebe17ce), (0x3939afe4), (0x4a4aed33),
	(0x4c4cff2b), (0x5858937b), (0xcfcf5b11),
	(0xd0d0066d), (0xefefbb91), (0xaaaa7b9e),
	(0xfbfbd7c1), (0x4343d217), (0x4d4df82f),
	(0x333399cc), (0x8585b622), (0x4545c00f),
	(0xf9f9d9c9), (0x02020e08), (0x7f7f66e7),
	(0x5050ab5b), (0x3c3cb4f0), (0x9f9ff04a),
	(0xa8a87596), (0x5151ac5f), (0xa3a344ba),
	(0x4040db1b), (0x8f8f800a), (0x9292d37e),
	(0x9d9dfe42), (0x3838a8e0), (0xf5f5fdf9),
	(0xbcbc19c6), (0xb6b62fee), (0xdada3045),
	(0x2121e784), (0x10107040), (0xffffcbd1),
	(0xf3f3efe1), (0xd2d20865), (0xcdcd5519),
	(0x0c0c2430), (0x1313794c), (0xececb29d),
	(0x5f5f8667), (0x9797c86a), (0x4444c70b),
	(0x1717655c), (0xc4c46a3d), (0xa7a758aa),
	(0x7e7e61e3), (0x3d3db3f4), (0x6464278b),
	(0x5d5d886f), (0x19194f64), (0x737342d7),
	(0x60603b9b), (0x8181aa32), (0x4f4ff627),
	(0xdcdc225d), (0x2222ee88), (0x2a2ad6a8),
	(0x9090dd76), (0x88889516), (0x4646c903),
	(0xeeeebc95), (0xb8b805d6), (0x14146c50),
	(0xdede2c55), (0x5e5e8163), (0x0b0b312c),
	(0xdbdb3741), (0xe0e096ad), (0x32329ec8),
	(0x3a3aa6e8), (0x0a0a3628), (0x4949e43f),
	(0x06061218), (0x2424fc90), (0x5c5c8f6b),
	(0xc2c27825), (0xd3d30f61), (0xacac6986),
	(0x62623593), (0x9191da72), (0x9595c662),
	(0xe4e48abd), (0x797974ff), (0xe7e783b1),
	(0xc8c84e0d), (0x373785dc), (0x6d6d18af),
	(0x8d8d8e02), (0xd5d51d79), (0x4e4ef123),
	(0xa9a97292), (0x6c6c1fab), (0x5656b943),
	(0xf4f4fafd), (0xeaeaa085), (0x6565208f),
	(0x7a7a7df3), (0xaeae678e), (0x08083820),
	(0xbaba0bde), (0x787873fb), (0x2525fb94),
	(0x2e2ecab8), (0x1c1c5470), (0xa6a65fae),
	(0xb4b421e6), (0xc6c66435), (0xe8e8ae8d),
	(0xdddd2559), (0x747457cb), (0x1f1f5d7c),
	(0x4b4bea37), (0xbdbd1ec2), (0x8b8b9c1a),
	(0x8a8a9b1e), (0x70704bdb), (0x3e3ebaf8),
	(0xb5b526e2), (0x66662983), (0x4848e33b),
	(0x0303090c), (0xf6f6f4f5), (0x0e0e2a38),
	(0x61613c9f), (0x35358bd4), (0x5757be47),
	(0xb9b902d2), (0x8686bf2e), (0xc1c17129),
	(0x1d1d5374), (0x9e9ef74e), (0xe1e191a9),
	(0xf8f8decd), (0x9898e556), (0x11117744),
	(0x696904bf), (0xd9d93949), (0x8e8e870e),
	(0x9494c166), (0x9b9bec5a), (0x1e1e5a78),
	(0x8787b82a), (0xe9e9a989), (0xcece5c15),
	(0x5555b04f), (0x2828d8a0), (0xdfdf2b51),
	(0x8c8c8906), (0xa1a14ab2), (0x89899212),
	(0x0d0d2334), (0xbfbf10ca), (0xe6e684b5),
	(0x4242d513), (0x686803bb), (0x4141dc1f),
	(0x9999e252), (0x2d2dc3b4), (0x0f0f2d3c),
	(0xb0b03df6), (0x5454b74b), (0xbbbb0cda),
	(0x16166258),
	(0x97636332), (0xeb7c7c6f), (0xc777775e),
	(0xf77b7b7a), (0xe5f2f2e8), (0xb76b6b0a),
	(0xa76f6f16), (0x39c5c56d), (0xc0303090),
	(0x04010107), (0x8767672e), (0xac2b2bd1),
	(0xd5fefecc), (0x71d7d713), (0x9aabab7c),
	(0xc3767659), (0x05caca40), (0x3e8282a3),
	(0x09c9c949), (0xef7d7d68), (0xc5fafad0),
	(0x7f595994), (0x074747ce), (0xedf0f0e6),
	(0x82adad6e), (0x7dd4d41a), (0xbea2a243),
	(0x8aafaf60), (0x469c9cf9), (0xa6a4a451),
	(0xd3727245), (0x2dc0c076), (0xeab7b728),
	(0xd9fdfdc5), (0x7a9393d4), (0x982626f2),
	(0xd8363682), (0xfc3f3fbd), (0xf1f7f7f3),
	(0x1dcccc52), (0xd034348c), (0xa2a5a556),
	(0xb9e5e58d), (0xe9f1f1e1), (0xdf71714c),
	(0x4dd8d83e), (0xc4313197), (0x5415156b),
	(0x1004041c), (0x31c7c763), (0x8c2323e9),
	(0x21c3c37f), (0x60181848), (0x6e9696cf),
	(0x1405051b), (0x5e9a9aeb), (0x1c070715),
	(0x4812127e), (0x368080ad), (0xa5e2e298),
	(0x81ebeba7), (0x9c2727f5), (0xfeb2b233),
	(0xcf757550), (0x2409093f), (0x3a8383a4),
	(0xb02c2cc4), (0x681a1a46), (0x6c1b1b41),
	(0xa36e6e11), (0x735a5a9d), (0xb6a0a04d),
	(0x535252a5), (0xec3b3ba1), (0x75d6d614),
	(0xfab3b334), (0xa42929df), (0xa1e3e39f),
	(0xbc2f2fcd), (0x268484b1), (0x575353a2),
	(0x69d1d101), (0x00000000), (0x99ededb5),
	(0x802020e0), (0xddfcfcc2), (0xf2b1b13a),
	(0x775b5b9a), (0xb36a6a0d), (0x01cbcb47),
	(0xcebebe17), (0xe43939af), (0x334a4aed),
	(0x2b4c4cff), (0x7b585893), (0x11cfcf5b),
	(0x6dd0d006), (0x91efefbb), (0x9eaaaa7b),
	(0xc1fbfbd7), (0x174343d2), (0x2f4d4df8),
	(0xcc333399), (0x228585b6), (0x0f4545c0),
	(0xc9f9f9d9), (0x0802020e), (0xe77f7f66),
	(0x5b5050ab), (0xf03c3cb4), (0x4a9f9ff0),
	(0x96a8a875), (0x5f5151ac), (0xbaa3a344),
	(0x1b4040db), (0x0a8f8f80), (0x7e9292d3),
	(0x429d9dfe), (0xe03838a8), (0xf9f5f5fd),
	(0xc6bcbc19), (0xeeb6b62f), (0x45dada30),
	(0x842121e7), (0x40101070), (0xd1ffffcb),
	(0xe1f3f3ef), (0x65d2d208), (0x19cdcd55),
	(0x300c0c24), (0x4c131379), (0x9dececb2),
	(0x675f5f86), (0x6a9797c8), (0x0b4444c7),
	(0x5c171765), (0x3dc4c46a), (0xaaa7a758),
	(0xe37e7e61), (0xf43d3db3), (0x8b646427),
	(0x6f5d5d88), (0x6419194f), (0xd7737342),
	(0x9b60603b), (0x328181aa), (0x274f4ff6),
	(0x5ddcdc22), (0x882222ee), (0xa82a2ad6),
	(0x769090dd), (0x16888895), (0x034646c9),
	(0x95eeeebc), (0xd6b8b805), (0x5014146c),
	(0x55dede2c), (0x635e5e81), (0x2c0b0b31),
	(0x41dbdb37), (0xade0e096), (0xc832329e),
	(0xe83a3aa6), (0x280a0a36), (0x3f4949e4),
	(0x18060612), (0x902424fc), (0x6b5c5c8f),
	(0x25c2c278), (0x61d3d30f), (0x86acac69),
	(0x93626235), (0x729191da), (0x629595c6),
	(0xbde4e48a), (0xff797974), (0xb1e7e783),
	(0x0dc8c84e), (0xdc373785), (0xaf6d6d18),
	(0x028d8d8e), (0x79d5d51d), (0x234e4ef1),
	(0x92a9a972), (0xab6c6c1f), (0x435656b9),
	(0xfdf4f4fa), (0x85eaeaa0), (0x8f656520),
	(0xf37a7a7d), (0x8eaeae67), (0x20080838),
	(0xdebaba0b), (0xfb787873), (0x942525fb),
	(0xb82e2eca), (0x701c1c54), (0xaea6a65f),
	(0xe6b4b421), (0x35c6c664), (0x8de8e8ae),
	(0x59dddd25), (0xcb747457), (0x7c1f1f5d),
	(0x374b4bea), (0xc2bdbd1e), (0x1a8b8b9c),
	(0x1e8a8a9b), (0xdb70704b), (0xf83e3eba),
	(0xe2b5b526), (0x83666629), (0x3b4848e3),
	(0x0c030309), (0xf5f6f6f4), (0x380e0e2a),
	(0x9f61613c), (0xd435358b), (0x475757be),
	(0xd2b9b902), (0x2e8686bf), (0x29c1c171),
	(0x741d1d53), (0x4e9e9ef7), (0xa9e1e191),
	(0xcdf8f8de), (0x569898e5), (0x44111177),
	(0xbf696904), (0x49d9d939), (0x0e8e8e87),
	(0x669494c1), (0x5a9b9bec), (0x781e1e5a),
	(0x2a8787b8), (0x89e9e9a9), (0x15cece5c),
	(0x4f5555b0), (0xa02828d8), (0x51dfdf2b),
	(0x068c8c89), (0xb2a1a14a), (0x12898992),
	(0x340d0d23), (0xcabfbf10), (0xb5e6e684),
	(0x134242d5), (0xbb686803), (0x1f4141dc),
	(0x529999e2), (0xb42d2dc3), (0x3c0f0f2d),
	(0xf6b0b03d), (0x4b5454b7), (0xdabbbb0c),
	(0x58161662),
	(0x32976363), (0x6feb7c7c), (0x5ec77777),
	(0x7af77b7b), (0xe8e5f2f2), (0x0ab76b6b),
	(0x16a76f6f), (0x6d39c5c5), (0x90c03030),
	(0x07040101), (0x2e876767), (0xd1ac2b2b),
	(0xccd5fefe), (0x1371d7d7), (0x7c9aabab),
	(0x59c37676), (0x4005caca), (0xa33e8282),
	(0x4909c9c9), (0x68ef7d7d), (0xd0c5fafa),
	(0x947f5959), (0xce074747), (0xe6edf0f0),
	(0x6e82adad), (0x1a7dd4d4), (0x43bea2a2),
	(0x608aafaf), (0xf9469c9c), (0x51a6a4a4),
	(0x45d37272), (0x762dc0c0), (0x28eab7b7),
	(0xc5d9fdfd), (0xd47a9393), (0xf2982626),
	(0x82d83636), (0xbdfc3f3f), (0xf3f1f7f7),
	(0x521dcccc), (0x8cd03434), (0x56a2a5a5),
	(0x8db9e5e5), (0xe1e9f1f1), (0x4cdf7171),
	(0x3e4dd8d8), (0x97c43131), (0x6b541515),
	(0x1c100404), (0x6331c7c7), (0xe98c2323),
	(0x7f21c3c3), (0x48601818), (0xcf6e9696),
	(0x1b140505), (0xeb5e9a9a), (0x151c0707),
	(0x7e481212), (0xad368080), (0x98a5e2e2),
	(0xa781ebeb), (0xf59c2727), (0x33feb2b2),
	(0x50cf7575), (0x3f240909), (0xa43a8383),
	(0xc4b02c2c), (0x46681a1a), (0x416c1b1b),
	(0x11a36e6e), (0x9d735a5a), (0x4db6a0a0),
	(0xa5535252), (0xa1ec3b3b), (0x1475d6d6),
	(0x34fab3b3), (0xdfa42929), (0x9fa1e3e3),
	(0xcdbc2f2f), (0xb1268484), (0xa2575353),
	(0x0169d1d1), (0x00000000), (0xb599eded),
	(0xe0802020), (0xc2ddfcfc), (0x3af2b1b1),
	(0x9a775b5b), (0x0db36a6a), (0x4701cbcb),
	(0x17cebebe), (0xafe43939), (0xed334a4a),
	(0xff2b4c4c), (0x937b5858), (0x5b11cfcf),
	(0x066dd0d0), (0xbb91efef), (0x7b9eaaaa),
	(0xd7c1fbfb), (0xd2174343), (0xf82f4d4d),
	(0x99cc3333), (0xb6228585), (0xc00f4545),
	(0xd9c9f9f9), (0x0e080202), (0x66e77f7f),
	(0xab5b5050), (0xb4f03c3c), (0xf04a9f9f),
	(0x7596a8a8), (0xac5f5151), (0x44baa3a3),
	(0xdb1b4040), (0x800a8f8f), (0xd37e9292),
	(0xfe429d9d), (0xa8e03838), (0xfdf9f5f5),
	(0x19c6bcbc), (0x2feeb6b6), (0x3045dada),
	(0xe7842121), (0x70401010), (0xcbd1ffff),
	(0xefe1f3f3), (0x0865d2d2), (0x5519cdcd),
	(0x24300c0c), (0x794c1313), (0xb29decec),
	(0x86675f5f), (0xc86a9797), (0xc70b4444),
	(0x655c1717), (0x6a3dc4c4), (0x58aaa7a7),
	(0x61e37e7e), (0xb3f43d3d), (0x278b6464),
	(0x886f5d5d), (0x4f641919), (0x42d77373),
	(0x3b9b6060), (0xaa328181), (0xf6274f4f),
	(0x225ddcdc), (0xee882222), (0xd6a82a2a),
	(0xdd769090), (0x95168888), (0xc9034646),
	(0xbc95eeee), (0x05d6b8b8), (0x6c501414),
	(0x2c55dede), (0x81635e5e), (0x312c0b0b),
	(0x3741dbdb), (0x96ade0e0), (0x9ec83232),
	(0xa6e83a3a), (0x36280a0a), (0xe43f4949),
	(0x12180606), (0xfc902424), (0x8f6b5c5c),
	(0x7825c2c2), (0x0f61d3d3), (0x6986acac),
	(0x35936262), (0xda729191), (0xc6629595),
	(0x8abde4e4), (0x74ff7979), (0x83b1e7e7),
	(0x4e0dc8c8), (0x85dc3737), (0x18af6d6d),
	(0x8e028d8d), (0x1d79d5d5), (0xf1234e4e),
	(0x7292a9a9), (0x1fab6c6c), (0xb9435656),
	(0xfafdf4f4), (0xa085eaea), (0x208f6565),
	(0x7df37a7a), (0x678eaeae), (0x38200808),
	(0x0bdebaba), (0x73fb7878), (0xfb942525),
	(0xcab82e2e), (0x54701c1c), (0x5faea6a6),
	(0x21e6b4b4), (0x6435c6c6), (0xae8de8e8),
	(0x2559dddd), (0x57cb7474), (0x5d7c1f1f),
	(0xea374b4b), (0x1ec2bdbd), (0x9c1a8b8b),
	(0x9b1e8a8a), (0x4bdb7070), (0xbaf83e3e),
	(0x26e2b5b5), (0x29836666), (0xe33b4848),
	(0x090c0303), (0xf4f5f6f6), (0x2a380e0e),
	(0x3c9f6161), (0x8bd43535), (0xbe475757),
	(0x02d2b9b9), (0xbf2e8686), (0x7129c1c1),
	(0x53741d1d), (0xf74e9e9e), (0x91a9e1e1),
	(0xdecdf8f8), (0xe5569898), (0x77441111),
	(0x04bf6969), (0x3949d9d9), (0x870e8e8e),
	(0xc1669494), (0xec5a9b9b), (0x5a781e1e),
	(0xb82a8787), (0xa989e9e9), (0x5c15cece),
	(0xb04f5555), (0xd8a02828), (0x2b51dfdf),
	(0x89068c8c), (0x4ab2a1a1), (0x92128989),
	(0x23340d0d), (0x10cabfbf), (0x84b5e6e6),
	(0xd5134242), (0x03bb6868), (0xdc1f4141),
	(0xe2529999), (0xc3b42d2d), (0x2d3c0f0f),
	(0x3df6b0b0), (0xb74b5454), (0x0cdabbbb),
	(0x62581616),
	(0x63329763), (0x7c6feb7c), (0x775ec777),
	(0x7b7af77b), (0xf2e8e5f2), (0x6b0ab76b),
	(0x6f16a76f), (0xc56d39c5), (0x3090c030),
	(0x01070401), (0x672e8767), (0x2bd1ac2b),
	(0xfeccd5fe), (0xd71371d7), (0xab7c9aab),
	(0x7659c376), (0xca4005ca), (0x82a33e82),
	(0xc94909c9), (0x7d68ef7d), (0xfad0c5fa),
	(0x59947f59), (0x47ce0747), (0xf0e6edf0),
	(0xad6e82ad), (0xd41a7dd4), (0xa243bea2),
	(0xaf608aaf), (0x9cf9469c), (0xa451a6a4),
	(0x7245d372), (0xc0762dc0), (0xb728eab7),
	(0xfdc5d9fd), (0x93d47a93), (0x26f29826),
	(0x3682d836), (0x3fbdfc3f), (0xf7f3f1f7),
	(0xcc521dcc), (0x348cd034), (0xa556a2a5),
	(0xe58db9e5), (0xf1e1e9f1), (0x714cdf71),
	(0xd83e4dd8), (0x3197c431), (0x156b5415),
	(0x041c1004), (0xc76331c7), (0x23e98c23),
	(0xc37f21c3), (0x18486018), (0x96cf6e96),
	(0x051b1405), (0x9aeb5e9a), (0x07151c07),
	(0x127e4812), (0x80ad3680), (0xe298a5e2),
	(0xeba781eb), (0x27f59c27), (0xb233feb2),
	(0x7550cf75), (0x093f2409), (0x83a43a83),
	(0x2cc4b02c), (0x1a46681a), (0x1b416c1b),
	(0x6e11a36e), (0x5a9d735a), (0xa04db6a0),
	(0x52a55352), (0x3ba1ec3b), (0xd61475d6),
	(0xb334fab3), (0x29dfa429), (0xe39fa1e3),
	(0x2fcdbc2f), (0x84b12684), (0x53a25753),
	(0xd10169d1), (0x00000000), (0xedb599ed),
	(0x20e08020), (0xfcc2ddfc), (0xb13af2b1),
	(0x5b9a775b), (0x6a0db36a), (0xcb4701cb),
	(0xbe17cebe), (0x39afe439), (0x4aed334a),
	(0x4cff2b4c), (0x58937b58), (0xcf5b11cf),
	(0xd0066dd0), (0xefbb91ef), (0xaa7b9eaa),
	(0xfbd7c1fb), (0x43d21743), (0x4df82f4d),
	(0x3399cc33), (0x85b62285), (0x45c00f45),
	(0xf9d9c9f9), (0x020e0802), (0x7f66e77f),
	(0x50ab5b50), (0x3cb4f03c), (0x9ff04a9f),
	(0xa87596a8), (0x51ac5f51), (0xa344baa3),
	(0x40db1b40), (0x8f800a8f), (0x92d37e92),
	(0x9dfe429d), (0x38a8e038), (0xf5fdf9f5),
	(0xbc19c6bc), (0xb62feeb6), (0xda3045da),
	(0x21e78421), (0x10704010), (0xffcbd1ff),
	(0xf3efe1f3), (0xd20865d2), (0xcd5519cd),
	(0x0c24300c), (0x13794c13), (0xecb29dec),
	(0x5f86675f), (0x97c86a97), (0x44c70b44),
	(0x17655c17), (0xc46a3dc4), (0xa758aaa7),
	(0x7e61e37e), (0x3db3f43d), (0x64278b64),
	(0x5d886f5d), (0x194f6419), (0x7342d773),
	(0x603b9b60), (0x81aa3281), (0x4ff6274f),
	(0xdc225ddc), (0x22ee8822), (0x2ad6a82a),
	(0x90dd7690), (0x88951688), (0x46c90346),
	(0xeebc95ee), (0xb805d6b8), (0x146c5014),
	(0xde2c55de), (0x5e81635e), (0x0b312c0b),
	(0xdb3741db), (0xe096ade0), (0x329ec832),
	(0x3aa6e83a), (0x0a36280a), (0x49e43f49),
	(0x06121806), (0x24fc9024), (0x5c8f6b5c),
	(0xc27825c2), (0xd30f61d3), (0xac6986ac),
	(0x62359362), (0x91da7291), (0x95c66295),
	(0xe48abde4), (0x7974ff79), (0xe783b1e7),
	(0xc84e0dc8), (0x3785dc37), (0x6d18af6d),
	(0x8d8e028d), (0xd51d79d5), (0x4ef1234e),
	(0xa97292a9), (0x6c1fab6c), (0x56b94356),
	(0xf4fafdf4), (0xeaa085ea), (0x65208f65),
	(0x7a7df37a), (0xae678eae), (0x08382008),
	(0xba0bdeba), (0x7873fb78), (0x25fb9425),
	(0x2ecab82e), (0x1c54701c), (0xa65faea6),
	(0xb421e6b4), (0xc66435c6), (0xe8ae8de8),
	(0xdd2559dd), (0x7457cb74), (0x1f5d7c1f),
	(0x4bea374b), (0xbd1ec2bd), (0x8b9c1a8b),
	(0x8a9b1e8a), (0x704bdb70), (0x3ebaf83e),
	(0xb526e2b5), (0x66298366), (0x48e33b48),
	(0x03090c03), (0xf6f4f5f6), (0x0e2a380e),
	(0x613c9f61), (0x358bd435), (0x57be4757),
	(0xb902d2b9), (0x86bf2e86), (0xc17129c1),
	(0x1d53741d), (0x9ef74e9e), (0xe191a9e1),
	(0xf8decdf8), (0x98e55698), (0x11774411),
	(0x6904bf69), (0xd93949d9), (0x8e870e8e),
	(0x94c16694), (0x9bec5a9b), (0x1e5a781e),
	(0x87b82a87), (0xe9a989e9), (0xce5c15ce),
	(0x55b04f55), (0x28d8a028), (0xdf2b51df),
	(0x8c89068c), (0xa14ab2a1), (0x89921289),
	(0x0d23340d), (0xbf10cabf), (0xe684b5e6),
	(0x42d51342), (0x6803bb68), (0x41dc1f41),
	(0x99e25299), (0x2dc3b42d), (0x0f2d3c0f),
	(0xb03df6b0), (0x54b74b54), (0xbb0cdabb),
	(0x16625816)
};

#endif


