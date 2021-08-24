#import<test.h>
@interface Test () {
    NSString * _something;
}
@end

@implementation Test
// @synthsize something = _something; 
- (NSString *) something { 
    return _something; 
    } 
- (void)setSomething:(NSString *)something { 
    _something = something; 
    }
@end