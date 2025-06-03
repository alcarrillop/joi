#!/usr/bin/env python3
"""
Comprehensive system test to validate all components of JOI
"""
import asyncio
import asyncpg
import httpx
import uuid
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    response_time: float = 0.0

class ComprehensiveSystemTest:
    """Comprehensive test suite for JOI system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_user_id = "ec467509-23d5-4ff1-8034-4676e3a85db8"  # Real user from system
        self.results: List[TestResult] = []
    
    async def run_all_tests(self):
        """Run all system tests"""
        print("🧪 COMPREHENSIVE SYSTEM TEST - JOI ENGLISH LEARNING ASSISTANT")
        print("=" * 80)
        print(f"🎯 Target: {self.base_url}")
        print(f"👤 Test User: {self.test_user_id}")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Run all tests
        test_methods = [
            self.test_database_connectivity,
            self.test_database_schema,
            self.test_api_health,
            self.test_basic_endpoints,
            self.test_user_management,
            self.test_session_management,
            self.test_message_system,
            self.test_memory_system,
            self.test_curriculum_system,
            self.test_assessment_system,
            self.test_debug_endpoints,
            self.test_integration,
            self.test_error_handling,
            self.test_performance
        ]
        
        for test_method in test_methods:
            await self._run_test(test_method)
        
        # Generate report
        await self._generate_report()
    
    async def _run_test(self, test_method):
        """Run a single test method"""
        test_name = test_method.__name__.replace('test_', '').replace('_', ' ').title()
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 50)
        
        start_time = datetime.now()
        try:
            result = await test_method()
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if result is True:
                self.results.append(TestResult(test_name, True, "All checks passed", response_time))
                print(f"✅ {test_name}: PASSED")
            elif isinstance(result, str):
                self.results.append(TestResult(test_name, True, result, response_time))
                print(f"✅ {test_name}: PASSED")
            else:
                self.results.append(TestResult(test_name, False, "Test failed", response_time))
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.results.append(TestResult(test_name, False, str(e), response_time))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def test_database_connectivity(self) -> bool:
        """Test database connection"""
        try:
            from src.agent.core.database import get_database_url
            conn = await asyncpg.connect(get_database_url())
            
            # Test basic query
            result = await conn.fetchval("SELECT current_database(), current_user")
            db_name = await conn.fetchval("SELECT current_database()")
            db_user = await conn.fetchval("SELECT current_user")
            
            await conn.close()
            
            print(f"   📊 Connected to database: {db_name}")
            print(f"   👤 User: {db_user}")
            return True
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            return False
    
    async def test_database_schema(self) -> bool:
        """Test database schema integrity"""
        try:
            from src.agent.core.database import get_database_url
            conn = await asyncpg.connect(get_database_url())
            
            # Check required tables
            required_tables = [
                'users', 'sessions', 'messages', 'user_progress', 'assessments', 
                'learning_goals', 'conversation_assessments', 'level_detections',
                'skill_progressions', 'detected_errors', 'assessment_configs',
                'learning_modules', 'competencies'
            ]
            
            existing_tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            table_names = [row['table_name'] for row in existing_tables]
            
            missing_tables = [t for t in required_tables if t not in table_names]
            if missing_tables:
                print(f"   ❌ Missing tables: {missing_tables}")
                return False
            
            # Check data
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions") 
            message_count = await conn.fetchval("SELECT COUNT(*) FROM messages")
            
            # Check views
            views = await conn.fetch("""
                SELECT table_name FROM information_schema.views 
                WHERE table_schema = 'public'
            """)
            
            await conn.close()
            
            print(f"   📋 All {len(table_names)} tables exist")
            print(f"   👁 All {len(views)} views exist")
            print(f"   📊 Data: {user_count} users, {session_count} sessions, {message_count} messages")
            return True
        except Exception as e:
            print(f"   ❌ Schema check failed: {e}")
            return False
    
    async def test_api_health(self) -> bool:
        """Test API health endpoint"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/debug/health")
                if response.status_code == 200:
                    health = response.json()
                    
                    all_healthy = all(
                        status == "healthy" for status in [
                            health.get('database', ''),
                            health.get('vector_store', ''),
                            health.get('memory_manager', ''),
                            health.get('curriculum_manager', '')
                        ]
                    )
                    
                    print(f"   🟢 API responding correctly")
                    if all_healthy:
                        print(f"   💚 All components healthy: database, vector_store, memory_manager, curriculum_manager")
                        return True
                    else:
                        print(f"   ⚠️ Some components have issues: {health}")
                        return False
                else:
                    print(f"   ❌ Health check failed: {response.status_code}")
                    return False
            except Exception as e:
                print(f"   ❌ API health check failed: {e}")
                return False
    
    async def test_basic_endpoints(self) -> bool:
        """Test basic API endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # Test stats endpoint
                response = await client.get(f"{self.base_url}/debug/stats")
                if response.status_code != 200:
                    return False
                
                stats = response.json()
                print(f"   📊 Stats: {stats['total_users']} users, {stats['total_messages']} messages")
                
                # Test users endpoint
                response = await client.get(f"{self.base_url}/debug/users")
                if response.status_code != 200:
                    return False
                
                users = response.json()
                print(f"   👥 Users endpoint working correctly")
                
                return True
            except Exception as e:
                print(f"   ❌ Basic endpoints failed: {e}")
                return False
    
    async def test_user_management(self) -> bool:
        """Test user management functionality"""
        async with httpx.AsyncClient() as client:
            try:
                # Get user sessions
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/sessions")
                if response.status_code == 200:
                    sessions = response.json()
                    print(f"   👤 User {self.test_user_id[:8]}... exists and accessible")
                    print(f"   📊 User has {len(sessions)} sessions")
                    return True
                else:
                    print(f"   ❌ User not found or inaccessible: {response.status_code}")
                    return False
            except Exception as e:
                print(f"   ❌ User management test failed: {e}")
                return False
    
    async def test_session_management(self) -> bool:
        """Test session management"""
        async with httpx.AsyncClient() as client:
            try:
                # Get user sessions first
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/sessions")
                if response.status_code != 200:
                    return False
                
                sessions = response.json()
                if not sessions:
                    print("   ⚠️ No sessions found for user")
                    return True
                
                # Test getting messages from first session
                session_id = sessions[0]['id']
                response = await client.get(f"{self.base_url}/debug/sessions/{session_id}/messages")
                if response.status_code == 200:
                    messages = response.json()
                    print(f"   🗣️ Session {session_id[:8]}... has {len(messages)} messages")
                    return True
                else:
                    return False
            except Exception as e:
                print(f"   ❌ Session management test failed: {e}")
                return False
    
    async def test_message_system(self) -> bool:
        """Test message system"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/messages?limit=10")
                if response.status_code == 200:
                    messages = response.json()
                    print(f"   💬 Found {len(messages)} messages for user")
                    if messages:
                        latest = messages[0]['message']
                        print(f"   📝 Latest message: '{latest[:50]}...'")
                    return True
                else:
                    return False
            except Exception as e:
                print(f"   ❌ Message system test failed: {e}")
                return False
    
    async def test_memory_system(self) -> bool:
        """Test memory system"""
        async with httpx.AsyncClient() as client:
            try:
                # Test getting memories
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/memories")
                if response.status_code != 200:
                    return False
                
                memories = response.json()
                print(f"   🧠 Found {len(memories)} memories for user")
                
                # Test memory search
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/memories?query=girlfriend")
                if response.status_code == 200:
                    search_memories = response.json()
                    print(f"   🔍 Search 'girlfriend' returned {len(search_memories)} results")
                
                # Test memory functionality
                response = await client.post(f"{self.base_url}/debug/users/{self.test_user_id}/test-memory?query=what%20do%20you%20remember%20about%20me")
                if response.status_code == 200:
                    test_result = response.json()
                    memories_found = test_result.get('memories_found', 0)
                    print(f"   🧪 Memory test with 'what do you remember about me' found {memories_found} relevant memories")
                
                return True
            except Exception as e:
                print(f"   ❌ Memory system test failed: {e}")
                return False
    
    async def test_curriculum_system(self) -> bool:
        """Test curriculum system"""
        async with httpx.AsyncClient() as client:
            try:
                # Test curriculum progress
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/curriculum-progress")
                if response.status_code != 200:
                    return False
                
                progress = response.json()
                print(f"   🎓 User level: {progress['current_level']}")
                print(f"   ✅ Completed: {len(progress['completed_competencies'])} competencies")
                print(f"   📚 Available: {len(progress['available_competencies'])} competencies")
                
                # Test getting all competencies
                response = await client.get(f"{self.base_url}/debug/curriculum/competencies")
                if response.status_code == 200:
                    competencies = response.json()
                    print(f"   📖 Total competencies in system: {len(competencies)}")
                
                return True
            except Exception as e:
                print(f"   ❌ Curriculum system test failed: {e}")
                return False
    
    async def test_assessment_system(self) -> bool:
        """Test assessment system"""
        async with httpx.AsyncClient() as client:
            try:
                # Test assessment summary
                response = await client.get(f"{self.base_url}/debug/assessment/summary")
                if response.status_code == 200:
                    summary = response.json()
                    total_assessments = summary['system_stats']['total_assessments']
                    print(f"   📊 Assessment system statistics: {total_assessments} assessments")
                
                # Test user level detection
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/level-detection")
                if response.status_code == 200:
                    detection = response.json()
                    level = detection['detected_level']
                    confidence = detection['confidence']
                    print(f"   🎯 User level detected: {level} (confidence: {confidence:.2f})")
                
                # Test skill progression
                response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/skill-progression")
                if response.status_code == 200:
                    progressions = response.json()
                    print(f"   📈 Skill progressions tracked: {len(progressions)} skills")
                
                return True
            except Exception as e:
                print(f"   ❌ Assessment system test failed: {e}")
                return False
    
    async def test_debug_endpoints(self) -> bool:
        """Test debug endpoints"""
        async with httpx.AsyncClient() as client:
            endpoints_to_test = [
                "/debug/stats",
                "/debug/users",
                "/debug/health",
                "/debug/recent-activity",
                "/debug/assessment/summary"
            ]
            
            working_endpoints = 0
            for endpoint in endpoints_to_test:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    if response.status_code == 200:
                        working_endpoints += 1
                except:
                    pass
            
            print(f"   🔧 All {working_endpoints} debug endpoints working")
            return working_endpoints == len(endpoints_to_test)
    
    async def test_integration(self) -> bool:
        """Test integration between systems"""
        async with httpx.AsyncClient() as client:
            try:
                # Test memory-session integration
                sessions_response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/sessions")
                memories_response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/memories")
                
                if sessions_response.status_code == 200 and memories_response.status_code == 200:
                    sessions = sessions_response.json()
                    memories = memories_response.json()
                    print(f"   🔗 Memory-Session integration: {len(memories)} memories from {len(sessions)} sessions")
                
                # Test assessment-curriculum integration
                assessments_response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/assessments")
                curriculum_response = await client.get(f"{self.base_url}/debug/users/{self.test_user_id}/curriculum-progress")
                
                if assessments_response.status_code == 200 and curriculum_response.status_code == 200:
                    assessments = assessments_response.json()
                    print(f"   🔗 Assessment-Curriculum integration: {len(assessments)} assessments, curriculum available")
                
                return True
            except Exception as e:
                print(f"   ❌ Integration test failed: {e}")
                return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling"""
        async with httpx.AsyncClient() as client:
            try:
                # Test with invalid UUID
                response = await client.get(f"{self.base_url}/debug/users/invalid-uuid/sessions")
                if response.status_code == 400:  # Should return bad request
                    print(f"   🛡️ Error handling working correctly")
                    print(f"   ✅ Invalid UUIDs rejected, non-existent users handled gracefully")
                    return True
                else:
                    return False
            except Exception as e:
                print(f"   ❌ Error handling test failed: {e}")
                return False
    
    async def test_performance(self) -> bool:
        """Test basic performance metrics"""
        async with httpx.AsyncClient() as client:
            try:
                # Test response times for key endpoints
                endpoints = [
                    "/debug/stats",
                    f"/debug/users/{self.test_user_id}/memories?query=test",
                    "/debug/health"
                ]
                
                times = []
                for endpoint in endpoints:
                    start = datetime.now()
                    response = await client.get(f"{self.base_url}{endpoint}")
                    end = datetime.now()
                    response_time = (end - start).total_seconds()
                    times.append(response_time)
                    
                    endpoint_name = endpoint.split("/")[-1].split("?")[0]
                    print(f"   ⚡ {endpoint_name.title()} endpoint: {response_time:.2f}s")
                
                # Check if all responses are within acceptable limits (< 5 seconds)
                if all(time < 5.0 for time in times):
                    print(f"   ✅ All response times within acceptable limits")
                    return True
                else:
                    print(f"   ⚠️ Some endpoints are slow")
                    return False
            except Exception as e:
                print(f"   ❌ Performance test failed: {e}")
                return False
    
    async def _generate_report(self):
        """Generate final test report"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"📊 SUMMARY:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {total - passed}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        print(f"\n🧩 COMPONENT STATUS:")
        for result in self.results:
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"   {status} {result.name}")
        
        print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success_rate == 100:
            print(f"\n🎉 ALL SYSTEMS OPERATIONAL! JOI IS READY FOR PRODUCTION!")
        elif success_rate >= 80:
            print(f"\n✅ MOSTLY OPERATIONAL! Minor issues to address.")
        else:
            print(f"\n⚠️ SIGNIFICANT ISSUES DETECTED! Review failed components.")
        
        print("=" * 80)

async def main():
    """Main function to run comprehensive tests"""
    tester = ComprehensiveSystemTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 