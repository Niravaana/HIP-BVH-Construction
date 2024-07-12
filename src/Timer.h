#pragma once

#include <dependencies/Orochi/Orochi/Orochi.h>
#include <src/Error.h>
#include <functional>
#include <unordered_map>

namespace BvhConstruction
{

class Timer final
{
  public:
	bool EnableTimer = true;

	using TokenType = int;
	using TimeUnit	= float;

	Timer() = default;

	Timer( const Timer& ) = default;
	Timer( Timer&& )	  = default;

	Timer& operator=( const Timer& )  = default;
	Timer& operator=( Timer&& other ) = default;

	~Timer() = default;

	class Profiler;

	template <typename CallableType, typename... Args>
	decltype( auto ) measure( const TokenType token, CallableType&& callable, Args&&... args ) noexcept
	{
		TimeUnit time{};
		oroEvent start{};
		oroEvent stop{};
		if ( EnableTimer )
		{
			CHECK_ORO( oroEventCreateWithFlags( &start, 0 ) );
			CHECK_ORO( oroEventCreateWithFlags( &stop, 0 ) );
			CHECK_ORO( oroEventRecord( start, 0 ) );
		}

		using return_type = std::invoke_result_t<CallableType, Args...>;
		if constexpr ( std::is_void_v<return_type> )
		{
			std::invoke( std::forward<CallableType>( callable ), std::forward<Args>( args )... );
			if ( EnableTimer )
			{
				CHECK_ORO( oroEventRecord( stop, 0 ) );
				CHECK_ORO( oroEventSynchronize( stop ) );
				CHECK_ORO( oroEventElapsedTime( &time, start, stop ) );
				CHECK_ORO( oroEventDestroy( start ) );
				CHECK_ORO( oroEventDestroy( stop ) );
				timeRecord[token] += time;
			}
			return;
		}
		else
		{
			decltype( auto ) result{ std::invoke( std::forward<CallableType>( callable ), std::forward<Args>( args )... ) };
			if ( EnableTimer )
			{
				CHECK_ORO( oroEventRecord( stop, 0 ) );
				CHECK_ORO( oroEventSynchronize( stop ) );
				CHECK_ORO( oroEventElapsedTime( &time, start, stop ) );
				CHECK_ORO( oroEventDestroy( start ) );
				CHECK_ORO( oroEventDestroy( stop ) );
				timeRecord[token] += time;
			}
			return result;
		}
	}

	[[nodiscard]] TimeUnit getTimeRecord( const TokenType token ) const noexcept
	{
		if ( timeRecord.find( token ) != timeRecord.end() ) return timeRecord.at( token );
		return TimeUnit{};
	}

	void reset( const TokenType token ) noexcept
	{
		if ( timeRecord.count( token ) > 0UL )
		{
			timeRecord[token] = TimeUnit{};
		}
	}

	void clear() noexcept { timeRecord.clear(); }

  private:
	using TimeRecord = std::unordered_map<TokenType, TimeUnit>;
	TimeRecord timeRecord;
};

} 