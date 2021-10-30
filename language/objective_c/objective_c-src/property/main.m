// #import<test.h>
// #import<Foundation/Foundation.h>

// int main(int argc, char * argv[]) {
//     Test *test = [[Test alloc] init];
//     test.something = @"hello";
//     NSLog(@"%@", test.something);
//     return 0;
// }


#import <Foundation/Foundation.h>
#include<stdio.h>

@interface Box:NSObject {
    double length;    // Length of a box
    double breadth;   // Breadth of a box
    double height;    // Height of a box
}

@property(nonatomic, readwrite) double height;  // Property
@property(nonatomic, readwrite) double length;
-(double) volume;
@end

@implementation Box

@synthesize height;
@synthesize length;
// -(double) length { return length; }
// -(void) setLength: (double) _value {length = _value; }

-(id)init {
    self = [super init];
    length = 1.0;
    breadth = 1.0;
    return self;
}

-(double) volume {
    return length*breadth*height;
}

@end

int main() {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];    
    Box *box1 = [[Box alloc]init];    // Create box1 object of type Box
    Box *box2 = [[Box alloc]init];    // Create box2 object of type Box

    double volume = 0.0;             // Store the volume of a box here

    // box 1 specification
    box1.height = 5.0; 

    // box 2 specification
    box2.height = 10.0;

    // volume of box 1
    volume = [box1 volume];
    NSLog(@"Volume of Box1 : %f", volume);

    // volume of box 2
    volume = [box2 volume];
    NSLog(@"Volume of Box2 : %f", volume);

    // int a = 1;
    // int b = ^{
    //     a = 2;
    //     a = 3;
    // };
    // printf("%d", b);


    [pool drain];
    return 0;
}