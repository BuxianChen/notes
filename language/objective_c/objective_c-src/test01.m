// compile command:
// gcc -o test01.exe test01.m -I c:/GNUstep/GNUstep/System/Library/Headers -L c:/GNUstep/GNUstep/System/Library/Libraries -std=c99 -lobjc -lgnustep-base -fconstant-string-class=NSConstantString

#import <Foundation/Foundation.h>

@interface SampleClass:NSObject
- (void) sampleMethod;
- (int) add:(int)a andB:(int)b;
@end

@implementation SampleClass
- (void) sampleMethod {
    NSLog(@"Hello, World! \n");
}

- (int) add:(int)a andB:(int)b {
	return a + b;
}
@end

int main() {
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
	SampleClass *sampleClass = [[SampleClass alloc]init];
	[sampleClass sampleMethod];
	int x = 1, y = 2;
	NSLog(@"%d + %d = %d", x, y, [sampleClass add:x andB:y]);
	[pool drain];
	return 0;
}