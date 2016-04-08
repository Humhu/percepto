#include "percepto/PerceptoUtils.h"

#include <iostream>
#include <stdexcept>

namespace percepto
{

void runtime_assert( bool test, const std::string& message )
{
#ifdef NO_ASSERTS
	return;
#else
	if( test ) { return; }
	std::cerr << message << std::endl;
	throw std::runtime_error( message );
#endif
}

}